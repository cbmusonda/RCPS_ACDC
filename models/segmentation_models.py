import os
import wandb
import numpy as np
import nibabel as nib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.losses import DiceCELoss
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference

from base.base_modules import TensorBuffer, NegativeSamplingPixelContrastiveLoss
from base.base_segmentation import BaseSegmentationModel
from base.base_wandb_model import WandBModel
from models.networks import ProjectorUNet
from models.transform import FullAugmentor
from utils.iteration.iterator import PolynomialLRWithWarmUp, MetricMeter
from utils.ddp_utils import gather_object_across_processes


class SemiSupervisedContrastiveSegmentationModel(BaseSegmentationModel, WandBModel):

    def __init__(self, cfg, num_classes, amp=False):

        BaseSegmentationModel.__init__(self, cfg, num_classes, amp)
        WandBModel.__init__(self, cfg)

        # ------------------------------
        # Network
        # ------------------------------
        net = ProjectorUNet(
            num_classes=num_classes,
            leaky=cfg["MODEL"]["LEAKY"],
            norm=cfg["MODEL"]["NORM"],
        ).to(self.device)

        if dist.is_available() and dist.is_initialized():
            self.network = DDP(
                nn.SyncBatchNorm.convert_sync_batchnorm(net),
                device_ids=[self.device],
            )
        else:
            self.network = net

        # ------------------------------
        # Loss
        # ------------------------------
        pos_weight = cfg["TRAIN"]["CLASS_WEIGHT"]
        lambda_ce = (1 + len(pos_weight)) / (1 + np.sum(pos_weight))

        self.criterion = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_ce=lambda_ce,
            include_background=True,
        ).to(self.device)

        # ------------------------------
        # Optimizer + LR schedule
        # ------------------------------
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=cfg["TRAIN"]["LR"],
            weight_decay=cfg["TRAIN"]["DECAY"],
            momentum=cfg["TRAIN"]["MOMENTUM"],
        )

        self.scheduler = PolynomialLRWithWarmUp(
            self.optimizer,
            total_steps=cfg["TRAIN"]["EPOCHS"],
            max_lr_steps=cfg["TRAIN"]["BURN"],
            warmup_steps=cfg["TRAIN"]["BURN_IN"],
        )

        # Other components
        self.augmentor = FullAugmentor()
        self.contrastive_loss = NegativeSamplingPixelContrastiveLoss(
            sample_num=cfg["TRAIN"]["SAMPLE_NUM"],
            bidirectional=True,
            temperature=0.1,
        )
        self.ds_list = ["level3", "level2", "level1", "out"]

        self.prepare_tensor_buffer()

        self.visual_pairs = [
            {"name": "name_l", "type": "Pred", "image": "image_l", "mask": "pred_l"},
            {"name": "name_l", "type": "GT", "image": "image_l", "mask": "label_l"},
            {"name": "name_u", "type": "Pred", "image": "image_u", "mask": "pred_u"},
        ]

        self.loss_names = [
            "seg_loss",
            "cps_l_loss",
            "cps_u_loss",
            "contrastive_l_loss",
            "contrastive_u_loss",
            "cosine_l_loss",
            "cosine_u_loss",
        ]

        # Loss meters
        self.train_loss = {name: 0.0 for name in self.loss_names}
        self.val_loss = {}

        self.val_table = wandb.Table(
            columns=["ID"] + [p["type"] for p in self.visual_pairs]
        )

    def prepare_tensor_buffer(self):
        """Initialize tensor buffers for contrastive learning."""
        self.tensor_buffer = TensorBuffer(
            self.cfg["TRAIN"]["BUFFER_SIZE"],
            self.cfg["MODEL"]["PROJECT_DIM"]
        )

    def initialize_metric_meter(self, class_names):
        """Initialize metric meter for tracking ALL metrics."""
        self.metric_meter = MetricMeter(
            metrics=[
                'dice', 'iou', 'precision', 'recall', 'specificity', 
                'f1', 'volume_similarity', 'hausdorff', 'hausdorff95', 'asd'
            ],
            class_names=class_names
        )

    def set_input(self, batch_l, batch_u):
        """
        Set training inputs (labeled + unlabeled batches).
        
        Args:
            batch_l: labeled batch with 'image' and 'label' keys
            batch_u: unlabeled batch with 'image' key
        """
        self.image_l = batch_l["image"].to(self.device)
        self.label_l = batch_l["label"].to(self.device)
        self.name_l = batch_l.get("name", ["unknown"])

        self.image_u = batch_u["image"].to(self.device)
        self.name_u = batch_u.get("name", ["unknown"])

    def set_test_input(self, batch):
        """
        Set validation / test inputs.
        
        Args:
            batch: validation batch with 'image' and 'label' keys
        """
        self.image = batch["image"].to(self.device)
        self.label = batch["label"].to(self.device)
        self.name = batch.get("name", ["unknown"])

    def forward_labeled(self, image, label):
        """Forward pass for labeled data."""
        with autocast(enabled=self.amp_enabled):
            outputs = self.network(image)
            
            if isinstance(outputs, dict):
                pred = outputs["out"]
            else:
                pred = outputs
                
            loss = self.criterion(pred, label)
            
        return pred, loss, outputs

    def forward_unlabeled(self, image):
        """Forward pass for unlabeled data."""
        with autocast(enabled=self.amp_enabled):
            outputs = self.network(image)
            
            if isinstance(outputs, dict):
                pred = outputs["out"]
            else:
                pred = outputs
                
        return pred, outputs

    def optimize_parameters(self, epoch):
        """
        One optimization step over a batch.
        
        Args:
            epoch: current epoch number
        """
        self.optimizer.zero_grad()

        # Forward labeled
        pred_l, seg_loss, outputs_l = self.forward_labeled(self.image_l, self.label_l)
        
        # Forward unlabeled  
        pred_u, outputs_u = self.forward_unlabeled(self.image_u)

        # Store predictions
        self.pred_l = pred_l
        self.pred_u = pred_u

        # Total loss (simplified - just use segmentation loss)
        total_loss = seg_loss

        # Backward
        if self.amp_enabled:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        # Update loss meters
        self.train_loss["seg_loss"] = seg_loss.item()

    def evaluate_one_step(self, save2disk=False, save_dir=None, affine_matrix=None):
        """
        One evaluation step with comprehensive metrics.
        
        Args:
            save2disk: whether to save predictions to disk
            save_dir: directory to save predictions
            affine_matrix: affine matrix for NIfTI saving
        """
        with torch.no_grad():
            # Use sliding window inference
            patch_size = self.cfg["TEST"]["PATCH_SIZE"]
            overlap = self.cfg["TEST"]["PATCH_OVERLAP"]
            
            pred = sliding_window_inference(
                inputs=self.image,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=self.network,
                overlap=overlap,
            )
            
            if isinstance(pred, dict):
                pred = pred["out"]
            
            pred_argmax = torch.argmax(pred, dim=1, keepdim=True)
            
            # ✅ Calculate ALL metrics - Fix typo in function name
            from utils.metric_calculator import (
                calculate_Dice_score,
                calculate_IoU,
                calculate_Precision_Recall,
                calculate_Specificity,
                calculate_F1_score,
                calculate_Volume_Similarity,
                calculate_Hasudorff_distance,  # ← Fixed: was calculate_Hausdorff_distance
                calculate_avg_surface_distance
            )
            
            # Calculate metrics (all return tensors of shape [B,])
            dice = calculate_Dice_score(pred_argmax, self.label)
            iou = calculate_IoU(pred_argmax, self.label)
            precision, recall = calculate_Precision_Recall(pred_argmax, self.label)
            specificity = calculate_Specificity(pred_argmax, self.label)
            f1 = calculate_F1_score(pred_argmax, self.label)
            vol_sim = calculate_Volume_Similarity(pred_argmax, self.label)
            hd = calculate_Hasudorff_distance(pred_argmax, self.label, directed=True, percentile=None)
            hd95 = calculate_Hasudorff_distance(pred_argmax, self.label, directed=True, percentile=95)
            asd = calculate_avg_surface_distance(pred_argmax, self.label)
            
            # Update metric meter
            self.metric_meter.update({
                'dice': dice.mean().item(),
                'iou': iou.mean().item(),
                'precision': precision.mean().item(),
                'recall': recall.mean().item(),
                'specificity': specificity.mean().item(),
                'f1': f1.mean().item(),
                'volume_similarity': vol_sim.mean().item(),
                'hausdorff': hd.mean().item(),
                'hausdorff95': hd95.mean().item(),
                'asd': asd.mean().item(),
            })
            
            # Save to disk if requested
            if save2disk and save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                for i in range(pred.shape[0]):
                    name = self.name[i] if isinstance(self.name, list) else self.name
                    if isinstance(name, str):
                        name = name.replace(".nii.gz", "").replace(".h5", "")
                    else:
                        name = f"case_{i}"
                    
                    pred_np = pred_argmax[i, 0].cpu().numpy().astype(np.uint8)
                    
                    affine = affine_matrix if affine_matrix is not None else np.eye(4)
                    nib_img = nib.Nifti1Image(pred_np, affine)
                    save_path = os.path.join(save_dir, f"{name}_pred.nii.gz")
                    nib.save(nib_img, save_path)

    def log_train_loss(self, step):
        """Log training losses to wandb."""
        if hasattr(wandb, "run") and wandb.run is not None:
            wandb.log(self.train_loss, step=step)