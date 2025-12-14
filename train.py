import wandb
import yaml
import tqdm
import argparse
import torch
import torch.distributed as dist
from itertools import cycle
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.transforms import *

from configs.experiment import prepare_experiment, update_config_file, makedirs
from models.segmentation_models import SemiSupervisedContrastiveSegmentationModel
from utils.iteration.load_data_v2 import TrainValDataPipeline
from utils.iteration.iterator import set_random_seed
from utils.ddp_utils import init_distributed_mode

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="la")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--pretrain_ckpt", type=str)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--ncpu", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--exp_name", type=str, default="running")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--entity", type=str)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--dist-url", default="env://")
    return parser.parse_args()


def main():
    args = parse_args()
    ngpu = torch.cuda.device_count()
    init_distributed_mode(args)

    print("------------------------------")
    print("Semi-Supervised Medical Image Segmentation Training")
    print(
        f"Mixed Precision - {args.mixed}; "
        f"CUDNN Benchmark - {args.benchmark}; "
        f"Num GPU - {ngpu}; Num Worker - {args.ncpu}"
    )

    # Load config
    cfg_file = f"configs/{args.task}.cfg"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    print("Loaded config:", cfg)

    cfg = update_config_file(args, cfg)

    seed = cfg["TRAIN"]["SEED"]
    batch_size = cfg["TRAIN"]["BATCHSIZE"]
    num_epochs = cfg["TRAIN"]["EPOCHS"]
    ratio = cfg["TRAIN"]["RATIO"]

    full_exp_name = f"{args.exp_name}-task_{args.task}-ratio_{ratio}"
    cfg["EXP_NAME"] = full_exp_name

    if args.debug:
        num_epochs = 2

    set_random_seed(seed=seed, benchmark=args.benchmark)

    train_aug = Compose(
        [
            LoadImaged(keys=["image", "label"], allow_missing_keys=True),
            EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
            NormalizeIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
            RandGridDistortiond(
                keys=["image", "label"],
                mode=["bilinear", "nearest"],
                distort_limit=0.1,
                device=torch.device("cuda"),
            ),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=cfg["TRAIN"]["PATCH_SIZE"],
                random_size=False,
            ),
        ]
    )

    val_aug = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    save_root = "./experiments"
    image_root, num_classes, class_names, affine = prepare_experiment(args.task)
    save_dir, metric_dir, infer_dir, vis_dir = makedirs(args.task, full_exp_name, save_root)

    data_pipeline = TrainValDataPipeline(image_root, label_ratio=ratio, random_seed=seed)
    trainset, unlabeled_set, valset = data_pipeline.get_dataset(
        train_aug, val_aug, cache_dataset=False
    )

    # ----------------------------------------------------------
    # FIX: COLAB NON-DISTRIBUTED MODE
    # ----------------------------------------------------------
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(trainset)
        unlabeled_sampler = DistributedSampler(unlabeled_set)
        val_sampler = DistributedSampler(valset)
    else:
        train_sampler = None
        unlabeled_sampler = None
        val_sampler = None

    print(
        f"Task {args.task} prepared. "
        f"Num labeled: {len(trainset)}; unlabeled: {len(unlabeled_set)}; val: {len(valset)}"
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=args.ncpu,
        sampler=train_sampler,
        shuffle=train_sampler is None,
    )
    unlabeled_loader = DataLoader(
        unlabeled_set,
        batch_size=batch_size,
        num_workers=args.ncpu,
        sampler=unlabeled_sampler,
        shuffle=unlabeled_sampler is None,
    )
    val_loader = DataLoader(
        valset,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
    )

    model = SemiSupervisedContrastiveSegmentationModel(
        cfg, num_classes=num_classes, amp=args.mixed
    )
    model.initialize_metric_meter(class_names)

    if args.pretrain_ckpt:
        model.load_networks(args.pretrain_ckpt, resume_training=False)

    if args.wandb and (not dist.is_initialized() or dist.get_rank() == 0):
        wandb.init(project=cfg["PROJECT"], name=cfg["EXP_NAME"], entity=args.entity)
        wandb.config.update(cfg)

    print("Start training...")

    # ------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------
    for epoch in range(model.start_epoch, num_epochs):

        if train_sampler:
            train_sampler.set_epoch(epoch)
        if unlabeled_sampler:
            unlabeled_sampler.set_epoch(epoch)

        model.train()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")

        iter_loader = iter(zip(cycle(train_loader), unlabeled_loader))
        _range = tqdm.tqdm(range(len(unlabeled_loader)), desc=f"Epoch {epoch+1}") if args.verbose else range(
            len(unlabeled_loader)
        )

        for _ in _range:
            batch_l, batch_u = next(iter_loader)
            model.set_input(batch_l, batch_u)
            model.optimize_parameters(epoch)

        if args.wandb and (not dist.is_initialized() or dist.get_rank() == 0):
            model.log_train_loss(step=epoch + 1)

        model.scheduler.step()

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            model.save_networks(epoch, save_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")

        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            model.metric_meter.reset()
            
            print(f"\nRunning validation at epoch {epoch+1}...")
            for batch in tqdm.tqdm(val_loader, desc="Validation") if args.verbose else val_loader:
                model.set_test_input(batch)
                model.evaluate_one_step(save2disk=True, save_dir=infer_dir, affine_matrix=affine)

            # ✅ Get ALL metrics
            stats = model.metric_meter.pop_mean_metric()
            
            # ✅ Print all metrics nicely formatted
            print(f"\n{'='*70}")
            print(f"Validation Results - Epoch {epoch+1}")
            print(f"{'='*70}")
            print(f"  Dice Score:          {stats.get('dice', 0.0):.4f}")
            print(f"  IoU (Jaccard):       {stats.get('iou', 0.0):.4f}")
            print(f"  Precision:           {stats.get('precision', 0.0):.4f}")
            print(f"  Recall (Sensitivity):{stats.get('recall', 0.0):.4f}")
            print(f"  Specificity:         {stats.get('specificity', 0.0):.4f}")
            print(f"  F1 Score:            {stats.get('f1', 0.0):.4f}")
            print(f"  Volume Similarity:   {stats.get('volume_similarity', 0.0):.4f}")
            print(f"  Hausdorff Distance:  {stats.get('hausdorff', 0.0):.2f} px")
            print(f"  Hausdorff 95:        {stats.get('hausdorff95', 0.0):.2f} px")
            print(f"  Avg Surface Distance:{stats.get('asd', 0.0):.2f} px")
            print(f"{'='*70}\n")
            
            # ✅ Log all metrics to W&B if enabled
            if args.wandb and (not dist.is_initialized() or dist.get_rank() == 0):
                wandb.log({
                    'val_dice': stats.get('dice', 0.0),
                    'val_iou': stats.get('iou', 0.0),
                    'val_precision': stats.get('precision', 0.0),
                    'val_recall': stats.get('recall', 0.0),
                    'val_specificity': stats.get('specificity', 0.0),
                    'val_f1': stats.get('f1', 0.0),
                    'val_volume_similarity': stats.get('volume_similarity', 0.0),
                    'val_hausdorff': stats.get('hausdorff', 0.0),
                    'val_hausdorff95': stats.get('hausdorff95', 0.0),
                    'val_asd': stats.get('asd', 0.0),
                }, step=epoch + 1)

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()