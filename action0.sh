# Preprocessing
# ./launchers/det_precompute.sh config/coordConv_B.py train
# ./launchers/det_precompute.sh config/coordConv_C.py train
# ./launchers/det_precompute.sh config/baseline_NA.py train
# ./launchers/det_precompute.sh config/NA_NLG.py train
# ./launchers/det_precompute.sh config/NA_NDIS.py train
# ./launchers/det_precompute.sh config/NA_NGF.py train
# ./launchers/det_precompute.sh config/NA_WGAC.py train
# ./launchers/det_precompute.sh config/NA_WGAC_cat.py train
# ./launchers/det_precompute.sh config/NA_WGAC_tmp.py train
# ./launchers/det_precompute.sh config/NA_WGAC_OFFSET0.py train
# ./launchers/det_precompute.sh config/NA_A.py train
# ./launchers/det_precompute.sh config/baseline.py train
./launchers/det_precompute.sh config/da_photo_dis.py train

# Training 
# ./launchers/train.sh  config/coordConv_A.py 0 coordConv_A
# ./launchers/train.sh  config/coordConv_B.py 1 coordConv_B
# ./launchers/train.sh  config/coordConv_C.py 0 coordConv_C
# ./launchers/train.sh  config/baseline_NA.py 0 baseline_NA
# ./launchers/train.sh  config/NA_NLG.py 1 NA_NLG
# ./launchers/train.sh  config/NA_NDIS.py 1 NA_NDIS
# ./launchers/train.sh  config/NA_NGF.py 0 NA_NGF
# ./launchers/train.sh  config/NA_WGAC.py 0 NA_WGAC
# ./launchers/train.sh  config/NA_WGAC_tmp.py 0 NA_WGAC_tmp
# ./launchers/train.sh  config/NA_WGAC_OFFSET0.py 0 NA_WGAC_OFFSET0
# ./launchers/train.sh  config/NA_A.py 0 NA_A
./launchers/train.sh config/da_photo_dis.py 1 da_photo_dis

# Evaluation on validation set
# ./launchers/det_precompute.sh config/tmp.py test
# ./launchers/eval.sh config/NA_WGAC.py 0 /home/lab530/KenYu/visualDet3D/NA_WGAC/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation

# Evaluation on testing set
# ./launchers/det_precompute.sh config/my_config.py test
# ./launchers/eval.sh config/my_config.py 0 /home/lab530/KenYu/visualDet3D/my_exp/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth test

