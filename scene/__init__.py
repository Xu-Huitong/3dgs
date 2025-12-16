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

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import BasicPointCloud
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # === 修改 1：将 ROI 存为成员变量 ===
        self.roi = scene_info.roi  # <--- 新增这一行
        # === 新增：ROI 相机剔除优化策略 ===
        if self.roi is not None:
            print(f"[INFO] Optimizing dataset: Filtering cameras based on ROI...")
            
            def is_camera_relevant(cam_info, roi):
                # 1. 获取相机位姿
                # 在 dataset_readers.py 中, cam_info.R 是 C2W 的旋转矩阵 (因为它是 W2C 的转置)
                # cam_info.T 是 W2C 的平移向量 (tvec)
                # 相机中心 World Center = -R * T
                cam_center = -np.dot(cam_info.R, cam_info.T)
                
                # 2. 向量计算：相机 -> ROI中心
                to_roi_vec = roi.center - cam_center
                dist_to_roi = np.linalg.norm(to_roi_vec)
                
                # 3. 策略A：距离保护
                # 如果相机在 ROI 内部或非常近（比如半径的 1.2 倍内），直接保留
                if dist_to_roi < roi.radius * 1.2:
                    return True
                
                # 4. 策略B：视场角过滤 (Frustum Culling)
                # cam_info.R 的第3列 (索引2) 是相机的 Z 轴 (Look At 方向)
                view_dir = cam_info.R[:, 2] 
                
                # 归一化方向向量
                to_roi_dir = to_roi_vec / dist_to_roi
                
                # 计算视线与物体方向的夹角余弦值 (Dot Product)
                cos_angle = np.dot(view_dir, to_roi_dir)
                
                # 阈值判断：
                # cos_angle > 0 表示视角在前方
                # cos_angle > 0.5 表示夹角小于 60 度，通常意味着物体在视野中心附近
                # 为了保险（避免切掉边缘视角），我们设置宽松一点，比如 0.2
                # cos_angle > 0.5（60度）甚至 0.7（45度），只保留正对着 ROI 的相机。
                if cos_angle > 0.4:
                    return True
                
                return False

            # 执行过滤
            original_count = len(scene_info.train_cameras)
            
            # 过滤训练集
            filtered_train = [c for c in scene_info.train_cameras if is_camera_relevant(c, self.roi)]
            # 过滤测试集
            filtered_test = [c for c in scene_info.test_cameras if is_camera_relevant(c, self.roi)]
            
            # 更新 scene_info (由于 NamedTuple 不可变，使用 _replace)
            scene_info = scene_info._replace(train_cameras=filtered_train, test_cameras=filtered_test)
            
            print(f"       Cameras optimized: {original_count} -> {len(filtered_train)} (Dropped {original_count - len(filtered_train)})")
            print(f"       [Performance] Skipped loading {original_count - len(filtered_train)} images!")
        # ===============================================
        if not self.loaded_iter:
            pcd = scene_info.point_cloud
            roi = scene_info.roi

            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            if roi is not None:
                print(f"[INFO] Filtering Point Cloud with ROI (OBB)...")
                print(f"       Original points: {pcd.points.shape[0]}")
                
                # === OBB 过滤逻辑 ===
                # 1. 坐标去中心化 (World -> Center relative)
                centered_points = pcd.points - roi.center
                
                # 2. 旋转到局部坐标系 (World -> Local)
                # 利用正交矩阵性质，乘以旋转矩阵相当于投影到主轴
                local_points = np.dot(centered_points, roi.rotation)
                
                # 3. 范围检查 (在局部坐标系下即为 AABB)
                # 检查 |x| <= extent_x 且 |y| <= extent_y ...
                mask = np.all(np.abs(local_points) <= roi.extent, axis=1)
                # ===================

                # 过滤点云
                filtered_pcd = BasicPointCloud(
                    points=pcd.points[mask],
                    colors=pcd.colors[mask],
                    normals=pcd.normals[mask]
                )
                print(f"       Filtered points: {filtered_pcd.points.shape[0]}")
                
                self.gaussians.create_from_pcd(filtered_pcd, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
