# Preprocessing
# ./launchers/det_precompute.sh config/data_augumentation/baseline.py train
# ./launchers/det_precompute.sh config/data_augumentation/viz_da.py train
# ./launchers/det_precompute.sh config/mixup/kitti_mixup_1.py train

# Training 
# ./launchers/train.sh config/mixup/kitti_mixup_1.py 0 kitti_mixup_1 > exp_output/mixup/kitti_mixup_1/screen_output.txt
./launchers/train.sh config/mixup/kitti_mixup_1.py 0 kitti_mixup_1

# Evaluation on validation set
# ./launchers/det_precompute.sh config/tmp.py test
# ./launchers/eval.sh config/NA_WGAC.py 0 /home/lab530/KenYu/visualDet3D/NA_WGAC/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation
# ./launchers/eval.sh config/nuscene_kitti.py 0 /home/lab530/KenYu/visualDet3D/exp_output/nuscene_kitti/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation

# Evaluation on testing set
# ./launchers/det_precompute.sh config/my_config.py test
# ./launchers/eval.sh config/my_config.py 0 /home/lab530/KenYu/visualDet3D/my_exp/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth test

