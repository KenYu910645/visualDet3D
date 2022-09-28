import numpy as np
import torch.nn as nn
import torch
import math
import time
from visualDet3D.networks.utils import DETECTOR_DICT
from visualDet3D.networks.detectors.yolomono3d_core import YoloMono3DCore
from visualDet3D.networks.heads.detection_3d_head import AnchorBasedDetection3DHead
from visualDet3D.networks.lib.blocks import AnchorFlatten
from visualDet3D.networks.lib.look_ground import LookGround
import pickle

class GroundAwareHead(AnchorBasedDetection3DHead):
    def init_layers(self, num_features_in,
                          num_anchors:int,
                          num_cls_output:int,
                          num_reg_output:int,
                          cls_feature_size:int=1024,
                          reg_feature_size:int=1024,
                          **kwargs):
        self.cls_feature_extraction = nn.Sequential(
            nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),

            nn.Conv2d(cls_feature_size, num_anchors*(num_cls_output), kernel_size=3, padding=1),
            AnchorFlatten(num_cls_output)
        )
        self.cls_feature_extraction[-2].weight.data.fill_(0)
        self.cls_feature_extraction[-2].bias.data.fill_(0)

        print(f"GroundAwareHead  self.exp = {self.exp}")
        if self.exp == "NA_NLG": # Without LookGround
            self.reg_feature_extraction = nn.Sequential(
                nn.Conv2d(num_features_in, reg_feature_size, 3, padding=1),
                nn.BatchNorm2d(reg_feature_size),
                nn.ReLU(),
                nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(reg_feature_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(reg_feature_size, num_anchors*num_reg_output, kernel_size=3, padding=1),
                AnchorFlatten(num_reg_output)
            )
        else:
            self.reg_feature_extraction = nn.Sequential(
                LookGround(num_features_in, self.exp),
                nn.Conv2d(num_features_in, reg_feature_size, 3, padding=1),
                nn.BatchNorm2d(reg_feature_size),
                nn.ReLU(),
                nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(reg_feature_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(reg_feature_size, num_anchors*num_reg_output, kernel_size=3, padding=1),
                AnchorFlatten(num_reg_output)
            )
        
        self.reg_feature_extraction[-2].weight.data.fill_(0)
        self.reg_feature_extraction[-2].bias.data.fill_(0)

    def forward(self, inputs):
        cls_preds = self.cls_feature_extraction(inputs['features'])
        if self.exp == "NA_NLG":
            reg_preds = self.reg_feature_extraction(inputs['features'])
        else:
            reg_preds = self.reg_feature_extraction(inputs)
        return cls_preds, reg_preds

@DETECTOR_DICT.register_module
class Yolo3D(nn.Module):
    """
        YoloMono3DNetwork
    """
    def __init__(self, network_cfg):
        super(Yolo3D, self).__init__()

        self.obj_types = network_cfg.obj_types
        
        self.exp = network_cfg.exp
        print(f"Yolo3D experiment setting = {self.exp}")
        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.core = YoloMono3DCore(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = AnchorBasedDetection3DHead(
            **(network_cfg.head)
        )

    def training_forward(self, img_batch, annotations, P2):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        
        # print(f"img_batch.shape = {img_batch.shape}") # [8, 3, 288, 1280]
        features  = self.core(dict(image=img_batch, P2=P2)) # [8, 1024, 18, 80]
       
        # For Experiment C
        if self.exp == "C":
            # print(f"features.shape = {features.shape}")
            grid = np.stack(self.build_tensor_grid([features.shape[2], features.shape[3]]), axis=0) #[2, h, w]
            grid = features.new(grid).unsqueeze(0).repeat(features.shape[0], 1, 1, 1) #[1, 2, h, w]
            features = torch.cat([features, grid], dim=1)
            # print(f"features.shape = {features.shape}")

        cls_preds, reg_preds = self.bbox_head(dict(features=features, P2=P2, image=img_batch))
        # print(f"cls_preds.shape = {cls_preds.shape}") # [8, 46080, 2]
        # print(f"reg_preds.shape = {reg_preds.shape}") # [8, 46080, 12] 
        anchors = self.bbox_head.get_anchor(img_batch, P2) # ([8, 1024, 18, 80])
        # print(f"anchors['anchors'] = {anchors['anchors'].shape}") # [1, 46080, 4]
        # print(f"anchors['mask'] = {anchors['mask'].shape}") # [8, 46080]
        # print(f"anchors['anchor_mean_std_3d'] = {anchors['anchor_mean_std_3d'].shape}") # [46080, 1, 6, 2], z, sin(\t), cos(\t)

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2)

        return cls_loss, reg_loss, loss_dict
    

    def build_tensor_grid(self, shape):
        """
            For CoordConv exp C
            input:
                shape = (h, w)
            output:
                yy_grid = (h, w)
                xx_grid = (h, w)
        """
        h, w = shape[0], shape[1]
        x_range = np.arange(h, dtype=np.float32)
        y_range = np.arange(w, dtype=np.float32)
        yy, xx  = np.meshgrid(y_range, x_range)
        yy_grid = 2.0 * yy / float(w) - 1 # Make sure value is [-1, 1]
        xx_grid = 2.0 * xx / float(h) - 1
        return yy_grid, xx_grid

    def test_forward(self, img_batch, P2):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            results: a nested list:
                result[i] = detection_results for obj_types[i]
                    each detection result is a list [scores, bbox, obj_type]:
                        bbox = [bbox2d(length=4) , cx, cy, z, w, h, l, alpha]
        """
        assert img_batch.shape[0] == 1 # we recommmend image batch size = 1 for testing
        
        # This is for visulization 
        # with open('img_batch.pkl', 'wb') as f:
        #     pickle.dump(img_batch, f)
        #     print(f"Write img_batch to img_batch.pkl")

        features  = self.core(dict(image=img_batch, P2=P2))

        # For experiment C
        if self.exp == "C":
            grid = np.stack(self.build_tensor_grid([features.shape[2], features.shape[3]]), axis=0) #[2, h, w]
            grid = features.new(grid).unsqueeze(0).repeat(features.shape[0], 1, 1, 1) #[1, 2, h, w]
            features = torch.cat([features, grid], dim=1)

        cls_preds, reg_preds = self.bbox_head(dict(features=features, P2=P2))

        anchors = self.bbox_head.get_anchor(img_batch, P2)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, img_batch)

        return scores, bboxes, cls_indexes

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) == 3:
            img_batch, annotations, calib = inputs
            return self.training_forward(img_batch, annotations, calib)
        else:
            img_batch, calib = inputs
            return self.test_forward(img_batch, calib)

@DETECTOR_DICT.register_module
class GroundAwareYolo3D(Yolo3D):
    """Some Information about GroundAwareYolo3D"""

    def build_head(self, network_cfg):
        self.bbox_head = GroundAwareHead(
            **(network_cfg.head)
        )

