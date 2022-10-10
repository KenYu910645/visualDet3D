# Preprocessing
./launchers/det_precompute.sh config/baseline.py train
./launchers/train.sh config/baseline.py 1 baseline > exp_output/baseline/screen_output.txt

# NMS
./launchers/det_precompute.sh config/nms_test/nms_0.py train
./launchers/train.sh config/nms_test/nms_0.py 1 baseline > exp_output/nms_test/nms_0/screen_output.txt
./launchers/det_precompute.sh config/nms_test/nms_0_25.py train
./launchers/train.sh config/nms_test/nms_0_25.py 1 baseline > exp_output/nms_test/nms_0_25/screen_output.txt
./launchers/det_precompute.sh config/nms_test/nms_0_5.py train
./launchers/train.sh config/nms_test/nms_0_5.py 1 baseline > exp_output/nms_test/nms_0_5/screen_output.txt
./launchers/det_precompute.sh config/nms_test/nms_0_75.py train
./launchers/train.sh config/nms_test/nms_0_75.py 1 baseline > exp_output/nms_test/nms_0_75/screen_output.txt

# Evaluation on validation set
# ./launchers/det_precompute.sh config/tmp.py test
# ./launchers/eval.sh config/NA_WGAC.py 0 /home/lab530/KenYu/visualDet3D/NA_WGAC/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth validation

# Evaluation on testing set
# ./launchers/det_precompute.sh config/my_config.py test
# ./launchers/eval.sh config/my_config.py 0 /home/lab530/KenYu/visualDet3D/my_exp/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth test

