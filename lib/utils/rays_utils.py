import torch
from torch.nn import functional as F
import ipdb

def raw2outputs(raw, z_vals, rays_d, white_bkgd=True):
    '''
    Transforms model's predictions to semantically meaningful values

    '''
    
    raw2alpha = lambda raw, dists, act_fn=F.relu : 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)      # 这个是真的距离

    rgb = torch.sigmoid(raw[..., :3])
    
    # NeRF 网络输出的 alpha 对应于论文里的sigma
    alpha = raw2alpha(raw[..., 3], dists)

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)   # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)

    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples):
    # pdf, cdf 均是关于 weights 的概率函数
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)    # [1024, 62]
    cdf = torch.cumsum(pdf, -1)                             # [1024, 62]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins)) [1024, 63]

    u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(weights.device)   # [1024, 128]

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)           # 查找合适的位置插入， right=True返回最后一个合适的位置
                                                            # cdf是被插入的， u是插入的
    below = torch.max(torch.zeros_like(inds-1), inds-1)        # [1024, 128]
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)  # [1024, 128]
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)   # [1024, 128, 2]

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]   # [1024, 128, 63]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)     # 通过inds，从cdf中选出相应的累积概率
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples