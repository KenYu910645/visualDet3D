
# EXP_NAME=('score_0_5' 'score_0_9')
EXP_NAME=('score_0_nms_0')

for exp_name in "${EXP_NAME[@]}"
do
    ./launchers/det_precompute.sh config/score_test/"$exp_name".py train
    ./launchers/train.sh config/score_test/"$exp_name".py 1 "$exp_name" > exp_output/score_test/"$exp_name"/screen_output.txt
done

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

# Evaluation on testing set
# ./launchers/det_precompute.sh config/my_config.py test
# ./launchers/eval.sh config/my_config.py 0 /home/lab530/KenYu/visualDet3D/my_exp/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth test

