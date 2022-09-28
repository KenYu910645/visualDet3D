import glob
import os
from shutil import rmtree

OUTPUT_DIR = "/home/lab530/KenYu/visualDet3D/my_exp/Mono3D/output/validation/data_rename/"

with open("/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/val.txt", 'r') as f:
    chen_val = [i.split('\n')[0] for i in f.readlines()]
print(f"chen split validation set size = {len(chen_val)}")

old_names = sorted(glob.glob("/home/lab530/KenYu/visualDet3D/my_exp/Mono3D/output/validation/data/*.txt"))
print(f"predict output file size = {len(old_names)}")

assert len(old_names) == len(chen_val)

# Clean output directory 
if os.path.exists(OUTPUT_DIR):
    rmtree(OUTPUT_DIR)
os.mkdir(OUTPUT_DIR)

for i, old_name in enumerate(old_names):
    new_name = OUTPUT_DIR + chen_val[i] + '.txt'
    os.rename(old_name, new_name)
    print(f"change file name from {old_name} -> {new_name}")
