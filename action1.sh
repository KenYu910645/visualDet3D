# Preprocessing
# ./launchers/det_precompute.sh config/coordConv_B.py train
# ./launchers/det_precompute.sh config/coordConv_C.py train
./launchers/det_precompute.sh config/baseline_NA.py train
# ./launchers/det_precompute.sh config/NA_NLG.py train
# ./launchers/det_precompute.sh config/NA_NDIS.py train
# ./launchers/det_precompute.sh config/NA_NGF.py train
# ./launchers/det_precompute.sh config/NA_WGAC.py train
# ./launchers/det_precompute.sh config/NA_WGAC_cat.py train
# ./launchers/det_precompute.sh config/NA_WGAC_tmp.py train
# ./launchers/det_precompute.sh config/NA_WGAC_OFFSET5.py train
# ./launchers/det_precompute.sh config/NA_B.py train
# Toy experiment
# ./launchers/det_precompute.sh config/toy_baseline.py train
# Data Augumentation test
# ./launchers/det_precompute.sh config/da_random_mirror.py train
# ./launchers/det_precompute.sh config/da_crop_top.py train
# ./launchers/det_precompute.sh config/da_no_right_img.py train

# Training 
# ./launchers/train.sh  config/coordConv_A.py 0 coordConv_A
# ./launchers/train.sh  config/coordConv_B.py 1 coordConv_B
# ./launchers/train.sh  config/coordConv_C.py 0 coordConv_C
./launchers/train.sh  config/baseline_NA.py 1 baseline_NA
# ./launchers/train.sh  config/NA_NLG.py 1 NA_NLG
# ./launchers/train.sh  config/NA_NDIS.py 1 NA_NDIS
# ./launchers/train.sh  config/NA_NGF.py 0 NA_NGF
# ./launchers/train.sh  config/NA_WGAC.py 0 NA_WGAC
# ./launchers/train.sh  config/NA_WGAC_tmp.py 0 NA_WGAC_tmp
# ./launchers/train.sh  config/NA_B.py 1 NA_B
# Toy experiment
# ./launchers/train.sh  config/toy_baseline.py 0 toy_baseline
# Data Augumentation test
# ./launchers/train.sh  config/da_random_mirror.py 1 da_random_mirror
# ./launchers/train.sh  config/da_crop_top.py 1 da_crop_top
# ./launchers/train.sh  config/da_no_right_img.py 1 da_no_right_img

# Evaluation on validation set
# ./launchers/det_precompute.sh config/tmp.py test
# ./launchers/eval.sh config/NA_WGAC.py 0 /home/lab530/KenYu/visualDet3D/NA_WGAC/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation
# ./launchers/eval.sh config/da_random_mirror.py 0 /home/lab530/KenYu/visualDet3D/toy_baseline_exp/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation

# Evaluation on testing set
# ./launchers/det_precompute.sh config/my_config.py test
# ./launchers/eval.sh config/my_config.py 0 /home/lab530/KenYu/visualDet3D/my_exp/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth test
