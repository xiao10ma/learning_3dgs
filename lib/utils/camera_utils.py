import torch
from torch import nn
import numpy as np
from lib.utils.general_utils import PILtoTorch
from lib.utils.graphics_utils import fov2focal, getWorld2View2, getProjectionMatrix
from tqdm import tqdm
from PIL import Image

WARNED = False

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        # self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_image = image.clamp(0.0, 1.0)

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.original_image *= gt_alpha_mask
        else:
            # self.original_image*= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.original_image*= torch.ones((1, self.image_height, self.image_width))

        # temp = self.original_image.permute(1, 2, 0)
        # temp = temp.numpy()
        # temp = Image.fromarray(np.array(temp*255.0, dtype=np.byte), "RGB")
        # temp.save('/data/duantong/mazipei/fic2.jpg')

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def to_dict(self):
        camera_dict = {
            'uid' : self.uid,
            'colmap_id' : self.colmap_id,
            'R' : self.R,
            'T' : self.T,
            'FoVx' : self.FoVx,
            'FoVy' : self.FoVy,
            'original_image' : self.original_image,
            'image_width' : self.image_width,
            'image_height' : self.image_height,
            'zfar' : self.zfar,
            'znear' : self.znear,
            'trans' : self.trans,
            'scale' : self.scale,
            'world_view_transform' : self.world_view_transform,
            'projection_matrix' : self.projection_matrix,
            'full_proj_transform' : self.full_proj_transform,
            'camera_center' : self.camera_center
        }
        return camera_dict

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # image_path = "/data/duantong/mazipei/fic.jpg"
    # cam_info.image.save(image_path)

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]

    temp = gt_image.permute(1, 2, 0)
    temp = temp.numpy()
    temp = Image.fromarray(np.array(temp*255.0, dtype=np.byte), "RGB")
    temp.save('/data/duantong/mazipei/fic3.jpg')

    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  uid=id, data_device=args.data_device) # uid是打乱后的从enumerate读的id，colmap_id是相机的id

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
        cam = loadCam(args, id, c, resolution_scale)
        cam = cam.to_dict()
        camera_list.append(cam)

    return camera_list

def camera_to_JSON(id, camera : Camera):
    # 这里的camera的旋转矩阵在第一次读的时候，经过转置了，现在再把它转置回来
    # camera.R，t 的是 W2C的
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    # W2C = np.linalg.inv(Rt)
    # 作者这里应该有错，我把它注释了，Rt本身已经是W2C了
    W2C = Rt
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]   # 转为列表
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }   # dict/JSON
    return camera_entry
