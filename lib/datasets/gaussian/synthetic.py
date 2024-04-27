import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2
from pathlib import Path
from PIL import Image
from typing import NamedTuple
from lib.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import tqdm
from lib.utils.camera_utils import cameraList_from_camInfos
import random
import ipdb

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    width: int
    height: int

def readCamerasFromTransfroms(path, transformfile, white_background, extension='.png'):
    cam_infos = []
    with open(os.path.join(path, transformfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents['camera_angle_x']
        
        frames = contents['frames']
        for idx, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            cam_name = os.path.join(path, frame['file_path'] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            # 此处的R，C是W2C的
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                                          # 在这里是第一次读cam的R，后续的都是经过了转置的
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            # image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array[0, 0, 0]

            norm_data = im_data / 255.
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, width=image.size[0], height=image.size[1]))

    return cam_infos            

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        # kwargs 是yaml文件中的train_dataset/test_dataset
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        # cams = kwargs['cams']
        self.data_root = os.path.join(data_root, scene)
        self.data_root = os.path.abspath(self.data_root)
        self.split = split
        white_bkgd = cfg.modelParam.white_bkgd
        resolution_scales = cfg.modelParam.resolution_scales


        if(self.split == 'train'):
            print('Reading Training Transforms')
            cam_infos = readCamerasFromTransfroms(self.data_root, "transforms_train.json", white_bkgd)
        else:
            print('Reading Test Transforms')
            cam_infos = readCamerasFromTransfroms(self.data_root, "transforms_test.json", white_bkgd)

        self.nerf_normalization = getNerfppNorm(cam_infos)     # translate = -center, radius = diagonal * 1.1 (diagonal 表示所有相机到相机平均位置的最大l2距离)

        random.shuffle(cam_infos)

        self.cameras = {}
        
        for resolution_scale in resolution_scales:
            if(self.split == 'train'):
                print('Loading Training Cameras')
            else:
                print('Loading Test Cameras')
            self.cameras[resolution_scale] = cameraList_from_camInfos(cam_infos, resolution_scale, cfg.modelParam)


    def __getitem__(self, index, scale=1.0):
        ret = self.cameras[scale][index]
        ret.update(self.nerf_normalization)
        return ret

    def __len__(self, scale=1.0):
        return len(self.cameras[scale])
