# ./launchers/det_precompute.sh config/anchor_gen.py train # test
# ./launchers/eval.sh config/anchor_gen.py 1 /home/lab530/KenYu/visualDet3D/exp_output/anchor_gen/Mono3D/checkpoint/BevAnkYolo3D_49.pth validation

# ./launchers/det_precompute.sh config/anchor_gen.py train
# ./launchers/train.sh config/anchor_gen.py 1 anchor_gen

# EXP_NAME=('anchor_gen_all_3Ddistance_bevAnchor_batch8' 'anchor_gen_all_L1distance_bevAnchor_batch1' 'anchor_gen_all_L1distance_bevAnchor_batch8' 'anchor_gen_all_maxIoU_gacAnchor_batch1' 'anchor_gen_all_maxIoU_gacAnchor_batch8')

EXP_NAME=('anchor_gen_all_gac_original' 'anchor_gen_dense_all_occluded_object' 'anchor_gen_dense_all_occluded_object')

for exp_name in "${EXP_NAME[@]}"
do
    ./launchers/det_precompute.sh config/anchor_gen/"$exp_name".py train
    ./launchers/train.sh config/anchor_gen/"$exp_name".py 1 "$exp_name"
done

# ./launchers/det_precompute.sh config/anchor_gen/anchor_gen_dense_all.py train
# ./launchers/train.sh config/anchor_gen/anchor_gen_dense_all.py 1 anchor_gen_dense_all

# NMS
# ./launchers/det_precompute.sh config/nms_test/nms_0.py train
# ./launchers/train.sh config/nms_test/nms_0.py 1 nms_0 > exp_output/nms_test/nms_0/screen_output.txt
# ./launchers/det_precompute.sh config/nms_test/nms_0_25.py train
# ./launchers/train.sh config/nms_test/nms_0_25.py 1 nms_0_25 > exp_output/nms_test/nms_0_25/screen_output.txt
# ./launchers/det_precompute.sh config/nms_test/nms_0_5.py train
# ./launchers/train.sh config/nms_test/nms_0_5.py 1 nms_0_5 > exp_output/nms_test/nms_0_5/screen_output.txt
# ./launchers/det_precompute.sh config/nms_test/nms_0_75.py train
# ./launchers/train.sh config/nms_test/nms_0_75.py 1 nms_0_75 > exp_output/nms_test/nms_0_75/screen_output.txt
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
# ./launchers/det_precompute.sh config/gac_original.py train
# ./launchers/train.sh config/gac_original.py 1 gac_original

# Evaluation on validation set
# ./launchers/det_precompute.sh config/anchor_gen.py train # test
# ./launchers/eval.sh config/anchor_gen.py 0 /home/lab530/KenYu/visualDet3D/exp_output/anchor_gen/Mono3D/checkpoint/BevAnkYolo3D_latest.pth validation
# python ../ml_toolkit/3d_object_detection_visualization/viz3Dbox.py

# ./launchers/det_precompute.sh config/anchor_gen.py train # test
# PTH_NAME=('BevAnkYolo3D_49.pth' 'BevAnkYolo3D_99.pth' 'BevAnkYolo3D_149.pth' 'BevAnkYolo3D_199.pth' 'BevAnkYolo3D_249.pth' 'BevAnkYolo3D_299.pth' 'BevAnkYolo3D_349.pth' 'BevAnkYolo3D_399.pth' 'BevAnkYolo3D_449.pth')
# for pth_name in "${PTH_NAME[@]}"
# do
#     ./launchers/eval.sh config/anchor_gen.py 1 /home/lab530/KenYu/visualDet3D/exp_output/anchor_gen/Mono3D/checkpoint/"$pth_name" validation > exp_output/anchor_gen/eval_val_result_"$pth_name".txt
# done


# Test one 
# ./launchers/det_precompute.sh config/anchor_gen.py test
# ./launchers/eval.sh config/anchor_gen.py 0 /home/lab530/KenYu/visualDet3D/exp_output/anchor_gen/Mono3D/checkpoint/BevAnkYolo3D_latest.pth test
# python ../ml_toolkit/3d_object_detection_visualization/viz3Dbox.py


