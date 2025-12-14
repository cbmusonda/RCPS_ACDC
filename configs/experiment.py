import os
from utils.iteration.load_data_v2 import simple_affine
from utils.ddp_utils import get_world_size


def prepare_experiment(task):
    if task == 'pancreas':
        # Pancreas path (leave as original, or change if you ever use it)
        image_root = './Pancreas'
        num_classes = 2
        class_names = ['bg', 'pancreas']
        affine = simple_affine

    elif task == 'la':
        # 🔴 THIS IS THE IMPORTANT LINE FOR YOUR PROJECT
        # RCPS-style LA dataset we created from mri_norm2.h5
        image_root = "/content/CML-main/CML-main/data/LA_rcps"

        # define task-specific information
        num_classes = 2
        class_names = ['bg', 'la']
        affine = simple_affine

    elif task == 'acdc':
        # RCPS-style ACDC dataset (same structure as LA_rcps)
        # ⬇️ change this path if your ACDC_rcps lives somewhere else
        image_root = "/content/CML-main/CML-main/data/ACDC_rcps"

        # ACDC usually has 4 classes: background + RV + MYO + LV
        num_classes = 4
        class_names = ['bg', 'RV', 'MYO', 'LV']
        affine = simple_affine
        
    else:
        raise NotImplementedError('Task {} not implemented'.format(task))

    return image_root, num_classes, class_names, affine


def makedirs(task, full_exp_name, save_root_path='./'):
    # NOTE: don't start the second argument to os.path.join with '/'
    # or it will ignore save_root_path.
    if task == 'pancreas':
        save_dir = os.path.join(save_root_path, 'checkpoints/pancreas/{}'.format(full_exp_name))
        os.makedirs(save_dir, exist_ok=True)

        metric_savedir = os.path.join(save_root_path, 'metrics/pancreas/{}'.format(full_exp_name))
        os.makedirs(metric_savedir, exist_ok=True)

        infer_save_dir = os.path.join(save_root_path, 'inference_display/pancreas/{}'.format(full_exp_name))
        os.makedirs(infer_save_dir, exist_ok=True)

        vis_save_dir = os.path.join(save_root_path, 'visualization/pancreas/{}'.format(full_exp_name))
        os.makedirs(vis_save_dir, exist_ok=True)

    elif task == 'la':
        save_dir = os.path.join(save_root_path, 'checkpoints/la/{}'.format(full_exp_name))
        os.makedirs(save_dir, exist_ok=True)

        metric_savedir = os.path.join(save_root_path, 'metrics/la/{}'.format(full_exp_name))
        os.makedirs(metric_savedir, exist_ok=True)

        infer_save_dir = os.path.join(save_root_path, 'inference_display/la/{}'.format(full_exp_name))
        os.makedirs(infer_save_dir, exist_ok=True)

        vis_save_dir = os.path.join(save_root_path, 'visualization/la/{}'.format(full_exp_name))
        os.makedirs(vis_save_dir, exist_ok=True)

    elif task == 'acdc':
        save_dir = os.path.join(save_root_path, 'checkpoints/acdc/{}'.format(full_exp_name))
        os.makedirs(save_dir, exist_ok=True)

        metric_savedir = os.path.join(save_root_path, 'metrics/acdc/{}'.format(full_exp_name))
        os.makedirs(metric_savedir, exist_ok=True)

        infer_save_dir = os.path.join(save_root_path, 'inference_display/acdc/{}'.format(full_exp_name))
        os.makedirs(infer_save_dir, exist_ok=True)

        vis_save_dir = os.path.join(save_root_path, 'visualization/acdc/{}'.format(full_exp_name))
        os.makedirs(vis_save_dir, exist_ok=True)
        
    else:
        raise NotImplementedError('Task {} not implemented'.format(task))

    return save_dir, metric_savedir, infer_save_dir, vis_save_dir


def update_config_file(args, cfg):
    cfg['TRAIN']['BURN_IN'] = cfg['TRAIN']['BURN_IN'] * get_world_size()
    cfg['TRAIN']['BURN'] = cfg['TRAIN']['BURN'] * get_world_size()
    cfg['TRAIN']['RAMPUP'] = cfg['TRAIN']['RAMPUP'] * get_world_size()
    cfg['TRAIN']['EPOCHS'] = cfg['TRAIN']['EPOCHS'] * get_world_size()

    if args.task == 'pancreas':
        cfg['TRAIN']['PATCH_SIZE'] = (96, 96, 96)
        cfg['TEST']['PATCH_SIZE'] = (96, 96, 96)
        cfg['TEST']['PATCH_OVERLAP'] = 7 / 8
        cfg['TRAIN']['CLASS_WEIGHT'] = [10.]
        cfg['TRAIN']['TEMP'] = 0.5
        cfg['PROJECT'] = 'SemiSegPancreas'

    if args.task == 'la':
        cfg['TRAIN']['PATCH_SIZE'] = (112, 112, 80)
        cfg['TEST']['PATCH_SIZE'] = (112, 112, 80)
        cfg['TEST']['PATCH_OVERLAP'] = 7 / 8
        cfg['TRAIN']['CLASS_WEIGHT'] = [10.]
        cfg['TRAIN']['TEMP'] = 0.5
        cfg['PROJECT'] = 'SemiSegLA'

    if args.task == 'acdc':
        # use the same patch size as LA to start with
        cfg['TRAIN']['PATCH_SIZE'] = (112, 112, 80)
        cfg['TEST']['PATCH_SIZE'] = (112, 112, 80)
        cfg['TEST']['PATCH_OVERLAP'] = 7 / 8
        cfg['TRAIN']['CLASS_WEIGHT'] = [10.]
        cfg['TRAIN']['TEMP'] = 0.5
        cfg['PROJECT'] = 'SemiSegACDC'
    return cfg