<<<<<<< HEAD
# Preprocessing
# ./launchers/det_precompute.sh config/data_augumentation/baseline.py train
# ./launchers/det_precompute.sh config/data_augumentation/viz_da.py train

# EXP_NAME=('kitti_seg_1') # ('kitti_seg_1' 'kitti_mixup_3')
# for exp_name in "${EXP_NAME[@]}"
# do
#     ./launchers/det_precompute.sh config/mixup/"$exp_name".py train
#     ./launchers/train.sh config/mixup/"$exp_name".py 0 "$exp_name" > exp_output/mixup/"$exp_name"/screen_output.txt
# done

# ./launchers/train.sh config/mixup/kitti_mixup_1.py 0 kitti_mixup_1

# /home/lab530/KenYu/visualDet3D/exp_output/mixup/kitti_mixup_1/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth

# Evaluation on validation set
# ./launchers/det_precompute.sh config/tmp.py test
# ./launchers/eval.sh config/NA_WGAC.py 0 /home/lab530/KenYu/visualDet3D/NA_WGAC/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation
# ./launchers/eval.sh config/nuscene_kitti.py 0 /home/lab530/KenYu/visualDet3D/exp_output/nuscene_kitti/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation

# Evaluation on testing set( a single image)
# ./launchers/det_precompute.sh config/mixup/kitti_mixup_1.py test
# ./launchers/eval.sh config/mixup/kitti_mixup_1.py 0 /home/lab530/KenYu/visualDet3D/exp_output/mixup/kitti_mixup_1/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth test
# python ../ml_toolkit/3d_object_detection_visualization/viz3Dbox.py


#########################
### Anchor generation ###
#########################

# Train for anchor gen experiment
./launchers/det_precompute.sh config/anchor_gen.py train
./launchers/train.sh config/anchor_gen.py 0 anchor_gen

# Evaluation on validation set
# ./launchers/det_precompute.sh config/anchor_gen.py train # test
# ./launchers/eval.sh config/anchor_gen.py 0 /home/lab530/KenYu/visualDet3D/exp_output/anchor_gen/Mono3D/checkpoint/BevAnkYolo3D_latest.pth validation
# python ../ml_toolkit/3d_object_detection_visualization/viz3Dbox.py

# Test one 
# ./launchers/det_precompute.sh config/anchor_gen.py test
# ./launchers/eval.sh config/anchor_gen.py 0 /home/lab530/KenYu/visualDet3D/exp_output/anchor_gen/Mono3D/checkpoint/BevAnkYolo3D_latest.pth test
# python ../ml_toolkit/3d_object_detection_visualization/viz3Dbox.py


