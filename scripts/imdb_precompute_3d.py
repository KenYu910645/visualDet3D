import numpy as np
import os
import pickle
import time
import cv2
from copy import deepcopy
import skimage.measure
import torch
import shutil

import matplotlib
matplotlib.use('agg') 

from _path_init import *
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou, BBox3dProjector
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.data.kitti.kittidata import KittiData, KittiLabel
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import cfg_from_file

import random 
import shutil
import copy
import json 
import numpy as np 
from math import sqrt
import argparse
import pickle

import sys
sys.path.insert(0, "/home/lab530/KenYu/ml_toolkit/kitti")
from iou_3d import get_3d_box, box3d_iou, box2d_iou, box2d_iog
from util_kitti import kitti_calib_file_parser, KITTI_Object

import sys
sys.path.insert(0, "/home/lab530/KenYu/ml_toolkit/data_augmentation/3Dmixup/")
from copy_paste import CopyPaste_Object

def process_train_val_file(cfg):
    train_file = cfg.data.train_split_file
    val_file   = cfg.data.val_split_file

    with open(train_file) as f:
        train_lines = f.readlines()
        for i  in range(len(train_lines)):
            train_lines[i] = train_lines[i].strip()

    with open(val_file) as f:
        val_lines = f.readlines()
        for i  in range(len(val_lines)):
            val_lines[i] = val_lines[i].strip()

    return train_lines, val_lines

def read_one_split(cfg, index_names, data_root_dir, output_dict, data_split = 'training', time_display_inter=100):
    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if data_split == 'training':
        disp_dir = os.path.join(save_dir, 'disp')
        if not os.path.isdir(disp_dir):
            os.mkdir(disp_dir)

    N = len(index_names)
    frames = [None] * N
    print("start reading {} data".format(data_split))
    timer = Timer()

    anchor_prior = getattr(cfg, 'anchor_prior', True)
    external_pixelwise_anchor = getattr(cfg.detector.anchors, 'external_pixelwise_anchor', "")

    total_objects        = [0 for _ in range(len(cfg.obj_types))]
    total_usable_objects = [0 for _ in range(len(cfg.obj_types))]
    if anchor_prior:
        anchor_manager = Anchors(cfg.path.preprocessed_path, readConfigFile=False, **cfg.detector.head.anchors_cfg)
        preprocess = build_augmentator(cfg.data.test_augmentation)
        total_objects        = [0 for _ in range(len(cfg.obj_types))]
        total_usable_objects = [0 for _ in range(len(cfg.obj_types))]
        
        len_scale = len(anchor_manager.scales)
        len_ratios = len(anchor_manager.ratios)
        len_level = len(anchor_manager.pyramid_levels)

        examine = np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios]) # [1, 16, 2]
        sums    = np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios, 3]) 
        squared = np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios, 3], dtype=np.float64)

        uniform_sum_each_type    = np.zeros((len(cfg.obj_types), 6), dtype=np.float64) # [z, sin2a, cos2a, w, h, l] : sum of all labels
        uniform_square_each_type = np.zeros((len(cfg.obj_types), 6), dtype=np.float64) # sqaure of all label

    is_copy_paste = False
    for d in cfg.data.train_augmentation:
        if d['type_name'] == 'CopyPaste':
            is_copy_paste = True
            copy_paste_use_seg      = d['keywords']['use_seg']
            copy_paste_solid_ratio  = d['keywords']['solid_ratio']
            copy_paste_scene_awaree = d['keywords']['use_scene_aware']
    
    print(f"[imdb_precompute_3d.py] is_copy_paste = {is_copy_paste}")
    instance_pool = []
    for i, index_name in enumerate(index_names):
        
        ######################################
        ### Build copy paste instance_pool ###
        ######################################
        if data_split == 'training' and is_copy_paste:
            P2   = kitti_calib_file_parser(os.path.join(data_root_dir, "calib", f"{index_name}.txt"))
            with open(os.path.join(data_root_dir, "label_2", f"{index_name}.txt")) as f:
                lines = f.read().splitlines()
                lines = list(lines for lines in lines if lines) # Delete empty lines
            objs =  [CopyPaste_Object(str_line + " NA",
                                      idx_img = index_name,
                                      idx_line = idx_line, 
                                      tf_matrix = P2) for idx_line, str_line in enumerate(lines)]
            # Filter inappropiate objs
            for obj in objs:
                if obj.category in cfg.obj_types and obj.truncated < 0.5 and obj.occluded == 0.0 and obj.area > 3000:
                    if copy_paste_use_seg and len(obj.seg_points) == 0 : continue
                    instance_pool.append(obj)
        
        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        calib, image, label, velo, depth = data_frame.read_data()
        
        # Load label , store the list of kittiObjet and kittiCalib, 

        if data_split == 'training':
            data_frame.label = [obj for obj in label.data if obj.type in cfg.obj_types and obj.occluded < cfg.data.max_occlusion and obj.z > cfg.data.min_z]
            
            if anchor_prior:
                for j in range(len(cfg.obj_types)):
                    total_objects[j] += len([obj for obj in data_frame.label if obj.type==cfg.obj_types[j]])
                    data = np.array(
                        [
                            [obj.z, np.sin(2*obj.alpha), np.cos(2*obj.alpha), obj.w, obj.h, obj.l]
                                for obj in data_frame.label if obj.type==cfg.obj_types[j]
                        ]
                    ) #[N, 6]
                    if data.any():
                        uniform_sum_each_type[j, :]    += np.sum(data, axis=0)
                        uniform_square_each_type[j, :] += np.sum(data ** 2, axis=0)
        else:
            data_frame.label = [obj for obj in label.data if obj.type in cfg.obj_types]
        
        # Load calibration file
        data_frame.calib = calib
        if data_split == 'training' and anchor_prior:
            original_image = image.copy()
            # baseline = (calib.P2[0, 3] - calib.P3[0, 3]) / calib.P2[0, 0]
            
            # Augument the images
            image, P2, label = preprocess(original_image, p2=deepcopy(calib.P2), labels=deepcopy(data_frame.label))
            _,  P3 = preprocess(original_image, p2=deepcopy(calib.P3))

            ## Computing statistic for positive anchors
            if len(data_frame.label) > 0:
                anchors, _ = anchor_manager(image[np.newaxis].transpose([0,3,1,2]), torch.tensor(P2).reshape([-1, 3, 4]))
                # print(f"[imdb_precomputer_3d.py] anchors = {anchors.shape}") # [1, 184320, 4]
                
                for j in range(len(cfg.obj_types)):
                    
                    # Label
                    bbox2d = torch.tensor([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in label if obj.type == cfg.obj_types[j]]).cuda()
                    if len(bbox2d) < 1:
                        continue
                    bbox3d = torch.tensor([[obj.x, obj.y, obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha)] for obj in label if obj.type == cfg.obj_types[j]]).cuda()

                    
                    usable_anchors = anchors[0]

                    IoUs = calc_iou(usable_anchors, bbox2d) #[N, K]
                    IoU_max, IoU_argmax = torch.max(IoUs, dim=0)
                    IoU_max_anchor, IoU_argmax_anchor = torch.max(IoUs, dim=1)

                    num_usable_object = torch.sum(IoU_max > cfg.detector.head.loss_cfg.fg_iou_threshold).item()
                    total_usable_objects[j] += num_usable_object

                    positive_anchors_mask = IoU_max_anchor > cfg.detector.head.loss_cfg.fg_iou_threshold
                    positive_ground_truth_3d = bbox3d[IoU_argmax_anchor[positive_anchors_mask]].cpu().numpy()

                    used_anchors = usable_anchors[positive_anchors_mask].cpu().numpy() #[x1, y1, x2, y2]

                    if external_pixelwise_anchor == "":
                        sizes_int, ratio_int = anchor_manager.anchors2indexes(used_anchors)
                        for k in range(len(sizes_int)):
                            examine[j, sizes_int[k], ratio_int[k]] += 1 # Denominator
                            sums[j, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5]
                            squared[j, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5] ** 2

        # Save label.txt and calib.txt in frames[ KittiData(), KittiData(), KittiData(), ...]
        frames[i] = data_frame

        if (i+1) % time_display_inter == 0:
            avg_time = timer.compute_avg_time(i+1)
            eta = timer.compute_eta(i+1, N)
            print("{} iter:{}/{}, avg-time:{}, eta:{}, total_objs:{}, usable_objs:{}".format(
                data_split, i+1, N, avg_time, eta, total_objects, total_usable_objects), end='\r')
    
    if data_split == 'training':
        print("\n")
        print(f"Best Possible Rate = {100*total_usable_objects[0]/total_objects[0]}%")
    
    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if data_split == 'training' and anchor_prior and external_pixelwise_anchor == "":
        
        for j in range(len(cfg.obj_types)):
            global_mean = uniform_sum_each_type[j] / total_objects[j]
            global_var  = np.sqrt(uniform_square_each_type[j] / total_objects[j] - global_mean ** 2)

            avg = sums[j] / (examine[j][:, :, np.newaxis] + 1e-8)
            EX_2 = squared[j] / (examine[j][:, :, np.newaxis] + 1e-8)
            std = np.sqrt(EX_2 - avg ** 2)

            # If data on this location is less than 10, ignore it's statistic
            avg[examine[j] < 10, :] = -100  # with such negative mean Z, anchors/losses will filter them out
            std[examine[j] < 10, :] = 1e10
            
            avg[np.isnan(std)]      = -100
            std[np.isnan(std)]      = 1e10
            avg[std < 1e-3]         = -100
            std[std < 1e-3]         = 1e10

            whl_avg = np.ones([avg.shape[0], avg.shape[1], 3]) * global_mean[3:6]
            whl_std = np.ones([avg.shape[0], avg.shape[1], 3]) * global_var[3:6]

            avg = np.concatenate([avg, whl_avg], axis=2)
            std = np.concatenate([std, whl_std], axis=2)

            cfg.data.anchor_mean_std_path     = getattr(cfg.data, 'anchor_mean_std_path', "/home/lab530/KenYu/visualDet3D/anchor/max_occlusion_2")
            cfg.data.is_overwrite_anchor_file = getattr(cfg.data, 'is_overwrite_anchor_file', False)
            cfg.data.is_use_anchor_file       = getattr(cfg.data, 'is_use_anchor_file', True)
            
            mean_file_dst = os.path.join(save_dir,'anchor_mean_{}.npy'.format(cfg.obj_types[j]))
            std_file_dst  = os.path.join(save_dir,'anchor_std_{}.npy'.format(cfg.obj_types[j]))
            if not cfg.data.is_use_anchor_file:
                # Use anchor that generate with this dataset
                np.save(mean_file_dst, avg)
                np.save(std_file_dst, std)
                if cfg.data.is_overwrite_anchor_file:
                    shutil.copyfile(mean_file_dst, cfg.data.anchor_mean_std_path + "_mean.npy")
                    shutil.copyfile(std_file_dst,  cfg.data.anchor_mean_std_path + "_std.npy")
                    print(f"Save anchor mean file to {cfg.data.anchor_mean_std_path}_mean.npy")
                    print(f"Save anchor std file to {cfg.data.anchor_mean_std_path}_std.npy")
            else:
                mean_file_src = cfg.data.anchor_mean_std_path + "_mean.npy"
                std_file_src  = cfg.data.anchor_mean_std_path + "_std.npy"
                shutil.copyfile(mean_file_src, mean_file_dst)
                shutil.copyfile(std_file_src,  std_file_dst)
                print(f"Using mean_file from {mean_file_src}")
                print(f"Using std_file from {std_file_src}")

    # For saving copy_paste instance pool
    if data_split == 'training' and is_copy_paste:
        
        # Save image source
        imgs_src  = {fn.split(".")[0]: cv2.imread(os.path.join(data_root_dir, "image_2", fn)) for fn in os.listdir(os.path.join(data_root_dir, "image_2"))}
        print(f"Number of source image in imgs_src: {len(imgs_src)}")
        pickle.dump(imgs_src, open(os.path.join(save_dir, "imgs_src.pkl"), "wb"))
        print(f"Saved source images to {os.path.join(save_dir, 'imgs_src.pkl')}")
        
        # Save instance pool
        print(f"Number of object in instance pool: {len(instance_pool)}")
        pickle.dump(instance_pool, open(os.path.join(save_dir, "instance_pool.pkl"), "wb"))
        print(f"Saved instance pool to {os.path.join(save_dir, 'instance_pool.pkl')}")
        
        
    pkl_file = os.path.join(save_dir,'imdb.pkl')
    pickle.dump(frames, open(pkl_file, 'wb'))
    print("{} split finished precomputing".format(data_split))

def main(config:str="config/config.py"):
    cfg = cfg_from_file(config)
    torch.cuda.set_device(cfg.trainer.gpu)
    
    # original 
    cfg.data.max_occlusion = getattr(cfg.data, 'max_occlusion', 2)
    cfg.data.min_z         = getattr(cfg.data, 'min_z', 3)
    print(f"max_occlusion = {cfg.data.max_occlusion}")
    print(f"min_z = {cfg.data.min_z}")
    
    is_copy_paste = any(d['type_name'] == 'CopyPaste' for d in cfg.data.train_augmentation)
    
    time_display_inter = 100 # define the inverval displaying time consumed in loop
    data_root_dir = cfg.path.data_path # the base directory of training dataset
    calib_path = os.path.join(data_root_dir, 'calib') 
    list_calib = os.listdir(calib_path)
    N = len(list_calib)
    
    # Load training datatset
    output_dict = {
                "calib": True,
                "image": True,
                "label": True,
                "velodyne": False,
                "depth": is_copy_paste,
            }
    train_names, val_names = process_train_val_file(cfg)
    read_one_split(cfg, train_names, data_root_dir, output_dict, 'training', time_display_inter)
    
    # Load validation datatset
    output_dict = {
                "calib": True,
                "image": False,
                "label": True,
                "velodyne": False,
                "depth": False,
            }
    read_one_split(cfg, val_names, data_root_dir, output_dict, 'validation', time_display_inter)

    print("Preprocessing finished")

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
