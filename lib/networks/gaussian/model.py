import torch
import torch.nn as nn
import numpy as np
from lib.networks.gaussian.gaussian_model import GaussianModel, BasicPointCloud
from lib.config import cfg
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from lib.utils.sh_utils import eval_sh
from lib.utils.sh_utils import SH2RGB
from plyfile import PlyData, PlyElement
import os

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T      # （100000, 3)
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0    # （100000, 3)
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T     # （100000, 3)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera['FoVx'] * 0.5)
    tanfovy = math.tan(viewpoint_camera['FoVy'] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera['image_height']),
        image_width=int(viewpoint_camera['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera['world_view_transform'],
        projmatrix=viewpoint_camera['full_proj_transform'],
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera['camera_center'],
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera['camera_center'].repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


class Model(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        self.gaussians = GaussianModel()
        
    def forward(self, batch):
        if batch['step'] == 1:
            self.loadply = True
            data_root = cfg.train_dataset.data_root
            data_root = os.path.join(data_root, cfg.scene)
            ply_path = os.path.join(data_root, "points3d.ply")
            if not os.path.exists(ply_path):
                # Since this data set has no colmap data, we start with random points
                num_pts = 100_000
                print(f"Generating random point cloud ({num_pts})...")
                
                # We create random points inside the bounds of the synthetic Blender scenes
                xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
                shs = np.random.random((num_pts, 3)) / 255.0
                pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

                storePly(ply_path, xyz, SH2RGB(shs) * 255)
            try:
                pcd = fetchPly(ply_path)
            except:
                pcd = None
            self.gaussians.create_from_pcd(pcd, batch['radius'])
            self.gaussians.training_setup(cfg.optParam)

        iteration = batch['step'] + 1
        
        self.gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        bg_color = [1, 1, 1] if cfg.modelParam.white_bkgd else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_pkg = render(batch, self.gaussians, cfg.pipeParam, bg)
        image, self.viewspace_point_tensor, self.visibility_filter, self.radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        self.camera_extent = batch['radius']

        return image
    
    def densify_prune_opt(self, iteration):
        iteration += 1
        if iteration < cfg.optParam.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.gaussians.max_radii2D[self.visibility_filter] = torch.max(self.gaussians.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
            self.gaussians.add_densification_stats(self.viewspace_point_tensor, self.visibility_filter)

            if iteration > cfg.optParam.densify_from_iter and iteration % cfg.optParam.densification_interval == 0:
                size_threshold = 20 if iteration > cfg.optParam.opacity_reset_interval else None
                # 每100个回合，做densify(caused by over construction || under construction)
                # 当到了opacity_reset_interval(3000)后，同样的，每100个interval，根据opacity，做prune
                self.gaussians.densify_and_prune(cfg.optParam.densify_grad_threshold, 0.005, self.camera_extent, size_threshold)
            
            if iteration % cfg.optParam.opacity_reset_interval == 0 or (cfg.modelParam.white_bkgd and iteration == cfg.optParam.densify_from_iter):
                self.gaussians.reset_opacity()