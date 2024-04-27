import torch
import torch.nn as nn
from lib.config import cfg
from lib.utils.loss_utils import l1_loss, ssim
import os
import imageio
from lib.utils import img_utils

# 优化器
class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net

    def forward(self, batch):
        image = self.net(batch)
        gt_image = batch['original_image']
        Ll1 = l1_loss(image, gt_image)
        loss_ssim = 1.0 - ssim(image, gt_image)
        loss = (1.0 - cfg.optParam.lambda_dssim) * Ll1 + cfg.optParam.lambda_dssim * loss_ssim
        
        scalar_stats = {}

        loss_Ll1 = Ll1
        scalar_stats.update({'l1_loss': loss_Ll1})
        scalar_stats.update({'ssim_loss': loss_ssim})

        # psnr一般只用与评估
        # color_loss.detach()了，返回一个新的张量，但是脱离了计算图
        # 所以color_loss本身还是会被反向传播更新的，但是其新生成的副本不会参与反向传播，从而psnr也不会参与到反向传播
        # psnr = -10. * torch.log(color_loss_fine.detach()) / \
        #         torch.log(torch.Tensor([10.]).to(color_loss_fine.device))    # device = cuda0
        # scalar_stats.update({'psnr': psnr})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return image, loss, scalar_stats, image_stats