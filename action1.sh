# Preprocessing
# ./launchers/det_precompute.sh config/data_augumentation/add_right_img.py train
# ./launchers/det_precompute.sh config/nuscene_kitti.py train

# Training 
# ./launchers/train.sh config/data_augumentation/add_right_img.py 1 add_right_img > exp_output/data_augumentation/add_right_img/screen_output.txt
# ./launchers/train.sh config/nuscene_kitti.py 0 nuscene_kitt


# Evaluation on validation set
# ./launchers/det_precompute.sh config/tmp.py test
# ./launchers/eval.sh config/NA_WGAC.py 0 /home/lab530/KenYu/visualDet3D/NA_WGAC/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation
./launchers/eval.sh config/nuscene_kitti.py 0 /home/lab530/KenYu/visualDet3D/exp_output/nuscene_kitti/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation

# Evaluation on testing set
# ./launchers/det_precompute.sh config/my_config.py test
# ./launchers/eval.sh config/my_config.py 0 /home/lab530/KenYu/visualDet3D/my_exp/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth test
