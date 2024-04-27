import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import torch
import lpips
import imageio
from lib.utils import img_utils
import cv2
import json
from PIL import Image

class Evaluator:

    def __init__(self,):
        self.psnrs = []
        os.system('mkdir -p ' + cfg.result_dir)
        os.system('mkdir -p ' + cfg.result_dir + '/vis')
        self.cnt = 0

    def evaluate(self, output, batch):
        # assert image number = 1
        pred_rgb = output.permute(1, 2, 0).detach().cpu().numpy()
        gt_rgb = torch.squeeze(batch['original_image'])
        gt_rgb = gt_rgb.permute(1, 2, 0).detach().cpu().numpy()
        psnr_item = psnr(gt_rgb, pred_rgb, data_range=1.)
        self.psnrs.append(psnr_item)
        save_path = os.path.join(cfg.result_dir, 'vis/res{:06d}.jpg'.format(self.cnt))

        self.cnt += 1

        save_img = img_utils.horizon_concate(gt_rgb, pred_rgb)
        save_img = Image.fromarray(np.array(save_img*255.0, dtype=np.byte), "RGB")
        save_img.save(save_path)

    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        print(ret)
        self.psnrs = []
        print('Save visualization results to {}'.format(cfg.result_dir))
        json.dump(ret, open(os.path.join(cfg.result_dir, 'metrics.json'), 'w'))
        return ret