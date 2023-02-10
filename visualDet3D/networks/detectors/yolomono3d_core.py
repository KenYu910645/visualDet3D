import numpy as np
import torch.nn as nn
import torch
import math
import time
from visualDet3D.networks.backbones import resnet
from visualDet3D.networks.detectors.my_coordconv import MyCoordConv

class YoloMono3DCore(nn.Module):
    """Some Information about YoloMono3DCore"""
    def __init__(self, backbone_arguments=dict()):
        super(YoloMono3DCore, self).__init__()
        self.exp = backbone_arguments.exp
        self.backbone = resnet(**backbone_arguments)
        self.coordconv = MyCoordConv()
    
    def forward(self, x):
        # print(f"x['image'] = {x['image'].shape}") # torch.Size([8, 3, 288, 1280])
        
        # Experiment Settings
        if self.exp == "A":
            x = self.coordconv(x['image'])
            x = self.backbone(x)
        elif self.exp == "B":
            self.grid = np.stack(self.build_tensor_grid([x['image'].shape[2], x['image'].shape[3]]), axis=0) #[2, h, w]
            self.grid = x['image'].new(self.grid).unsqueeze(0).repeat(x['image'].shape[0], 1, 1, 1) #[1, 2, h, w]
            x = torch.cat([x['image'], self.grid], dim=1)
            x = self.backbone(x)
        else: # "baseline" or "C"
            # Without Coordconv/ CoordConv_C
            x = self.backbone(x['image'])
        
        x = x[0]
        # print(f"x.shape = {x.shape}") # torch.Size([8, 1024, 18, 80]
        return x
    
    def build_tensor_grid(self, shape):
        """
            For CoordConv exp B
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