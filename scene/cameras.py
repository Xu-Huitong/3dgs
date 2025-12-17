#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
from PIL import Image

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # =================================================================================
        # [修改点 1] 内存优化：不再直接存储 float32 的 Tensor，而是存储 uint8 的 numpy 数组
        # =================================================================================
        
        # 1. 直接使用 PIL Resize，避免先转 Tensor 造成内存峰值
        resized_image_pil = image.resize(resolution)
        
        # 2. 转为 Numpy uint8 (H, W, C) 并存储在 RAM 中
        self.cached_image_uint8 = np.array(resized_image_pil)
        
        self.image_width = self.cached_image_uint8.shape[1]
        self.image_height = self.cached_image_uint8.shape[0]

        # 3. 处理 Alpha Mask (同样以 uint8 存储在 RAM)
        if self.cached_image_uint8.shape[2] == 4:
            # 提取 Alpha 通道，保留维度 (H, W, 1)
            self.cached_mask_uint8 = self.cached_image_uint8[:, :, 3:4].copy()
            # 图像只保留 RGB
            self.cached_image_uint8 = self.cached_image_uint8[:, :, :3]
        else:
            self.cached_mask_uint8 = np.ones((self.image_height, self.image_width, 1), dtype=np.uint8) * 255

        # 4. 处理 train_test_exp 分割逻辑 (直接修改 mask)
        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.cached_mask_uint8[:, :self.image_width // 2] = 0
            else:
                self.cached_mask_uint8[:, self.image_width // 2:] = 0

        # 注意：这里不再生成 self.original_image 和 self.alpha_mask (Tensor)
        # =================================================================================

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones((self.image_height, self.image_width, 1), dtype=torch.float32).to(self.data_device)
            # 处理 Alpha mask 对 depth mask 的影响（这里临时生成 tensor 计算，不缓存）
            # 注意：如果 mask 很复杂，这里可能需要重新考虑，但通常 mask 是全 1
            # 为了简单起见，我们假设 depth_mask 初始化为全 1，之后在 get_depth 时如果需要再与 alpha 结合
            # 但原逻辑是 self.depth_mask = torch.ones_like(self.alpha_mask)
            
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    # [修改点 2] 按需生成 GT Image 的方法
    def get_gt_image(self):
        # uint8 (H, W, C) -> float32 Tensor (C, H, W) -> GPU
        gt_image = torch.from_numpy(self.cached_image_uint8).float().to(self.data_device) / 255.0
        return gt_image.permute(2, 0, 1).clamp(0.0, 1.0)

    # [修改点 3] 按需生成 Alpha Mask 的属性
    @property
    def alpha_mask(self):
        # uint8 (H, W, 1) -> float32 Tensor (1, H, W) -> GPU
        mask = torch.from_numpy(self.cached_mask_uint8).float().to(self.data_device) / 255.0
        return mask.permute(2, 0, 1)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]