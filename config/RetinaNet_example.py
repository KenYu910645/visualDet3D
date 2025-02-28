from easydict import EasyDict as edict
import os 
import numpy as np

cfg = edict()
cfg.obj_types = ['Car', 'Pedestrian', 'Cyclist']
cfg.anchor_prior  = False
cfg.exp = 'baseline'

## trainer
trainer = edict(
    gpu = 0,
    max_epochs = 15,
    disp_iter = 50,
    save_iter = 5,
    test_iter = 5,
    training_func = "train_mono_detection",
    test_func = "test_mono_detection",
    evaluate_func = "evaluate_kitti_obj",
)

cfg.trainer = trainer

## path
path = edict()
path.data_path = '/home/lab530/KenYu/kitti/training'# "/data/kitti_obj/training" # used in visualDet3D/data/.../dataset
path.test_path = '/home/lab530/KenYu/kitti/testing' # ""
path.visualDet3D_path = '/home/lab530/KenYu/visualDet3D/visualDet3D' # "/path/to/visualDet3D/visualDet3D" # The path should point to the inner subfolder
path.project_path = '/home/lab530/KenYu/visualDet3D/exp_output/retinaNet' # "/path/to/visualDet3D/workdirs" # or other path for pickle files, checkpoints, tensorboard logging and output files.

if not os.path.isdir(path.project_path):
    os.mkdir(path.project_path)
path.project_path = os.path.join(path.project_path, 'RetinaNet')
if not os.path.isdir(path.project_path):
    os.mkdir(path.project_path)

path.log_path = os.path.join(path.project_path, "log")
if not os.path.isdir(path.log_path):
    os.mkdir(path.log_path)

path.checkpoint_path = os.path.join(path.project_path, "checkpoint")
if not os.path.isdir(path.checkpoint_path):
    os.mkdir(path.checkpoint_path)

path.preprocessed_path = os.path.join(path.project_path, "output")
if not os.path.isdir(path.preprocessed_path):
    os.mkdir(path.preprocessed_path)

path.train_imdb_path = os.path.join(path.preprocessed_path, "training")
if not os.path.isdir(path.train_imdb_path):
    os.mkdir(path.train_imdb_path)

path.val_imdb_path = os.path.join(path.preprocessed_path, "validation")
if not os.path.isdir(path.val_imdb_path):
    os.mkdir(path.val_imdb_path)

cfg.path = path

## optimizer
optimizer = edict(
    type_name = 'adam',
    keywords = edict(
        lr        = 1e-4,
        weight_decay = 0,
    ),
    clipped_gradient_norm = 1.0
)
cfg.optimizer = optimizer
## scheduler
scheduler = edict(
    type_name = 'CosineAnnealingLR',
    keywords = edict(
        T_max     = cfg.trainer.max_epochs,
        eta_min   = 3e-5,
    )
)
cfg.scheduler = scheduler

## data
data = edict(
    batch_size = 8,
    num_workers = 8,
    rgb_shape = (288, 1280, 3),
    is_reproject = False,
    use_right_image = False,
    train_dataset = "KittiMonoDataset",
    val_dataset   = "KittiMonoDataset",
    test_dataset  = "KittiMonoTestDataset",
    train_split_file = os.path.join(cfg.path.visualDet3D_path, 'data', 'kitti', 'chen_split', 'train.txt'),
    val_split_file   = os.path.join(cfg.path.visualDet3D_path, 'data', 'kitti', 'chen_split', 'val.txt'),
)

data.augmentation = edict(
    rgb_mean = np.array([0.485, 0.456, 0.406]),
    rgb_std  = np.array([0.229, 0.224, 0.225]),
    cropSize = (data.rgb_shape[0], data.rgb_shape[1]),
    crop_top = 100,
    max_occlusion = 3,
    min_z    = 0
)
data.train_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='PhotometricDistort', keywords=edict(distort_prob=1.0, contrast_lower=0.5, contrast_upper=1.5, saturation_lower=0.5, saturation_upper=1.5, hue_delta=18.0, brightness_delta=32)),
    edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='RandomMirror', keywords=edict(mirror_prob=0.5)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
data.test_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
cfg.data = data

## networks
detector = edict()
detector.obj_types = cfg.obj_types
detector.name = 'RetinaNet'
detector.backbone = edict(
    depth=50,
    pretrained=True,
    frozen_stages=1,
    num_stages=4,
    out_indices=(1, 2, 3), #8, 16, 32
    norm_eval=True,
)
detector.neck  = edict(
    in_channels=[512, 1024, 2048], #only use 8 16 32
    out_channels=256,
    num_outs=5
)

anchors = edict(
    {
        'pyramid_levels':[i for i in range(3, 8)],   # [3,  4,  5,   6,   7]
        'strides': [2 ** (i) for i in range(3, 8)],  # [8,  16, 32,  64,  128]
        'sizes' : [4 * 2 ** i for i in range(3, 8)], # [32, 64, 128, 256, 512]
        'ratios': np.array([0.5, 1, 2.0]),
        'scales': np.array([2 ** (i / 3.0) for i in range(3)]), # [1, 1.26, 1,587]
    }
)
head_loss = edict(
    fg_iou_threshold = 0.5,
    bg_iou_threshold = 0.4,
    min_iou_threshold= 0,
    gamma = 2.0,
    balance_weights = [1],
    pos_weight       = -1,
)
head_test = edict(
    nms_pre=1000,
    score_thr=0.2,
    cls_agnostic = False,
    nms_iou_thr=0.4,
)
detector.head = edict(
    stacked_convs   = 4,
    in_channels     = 256,
    feat_channels   = 256,
    num_classes     = len(cfg.obj_types),
    target_stds     = [1.0, 1.0, 1.0, 1.0],
    target_means    = [ .0,  .0,  .0,  .0],
    anchors_cfg     = anchors,
    loss_cfg        = head_loss,
    test_cfg        = head_test
)
detector.loss = head_loss
cfg.detector = detector
