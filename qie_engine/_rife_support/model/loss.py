"""The two modules `train_log/RIFE_HDv3.py` actually instantiates, trimmed from RIFE's loss.py.

Upstream (r3gm/RIFE, Apache-2.0) also defines VGGPerceptualLoss and MeanShift; those pull in
`torchvision.models` and are training-only, so they are dropped here. `from model.loss import *`
in the RIFE archive still resolves, and the import no longer drags torchvision into the path.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return loss_map * loss_mask


class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        return transf / torch.sqrt(0.81 + transf ** 2)

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        return torch.mean(dist / (0.1 + dist), 1, True)

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        return F.pad(inner, [padding] * 4)

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, pred, gt):
        n, c, h, w = pred.shape
        stack = torch.cat([pred.reshape(n * c, 1, h, w), gt.reshape(n * c, 1, h, w)], 0)
        stack_x = F.conv2d(stack, self.kernelX, padding=1)
        stack_y = F.conv2d(stack, self.kernelY, padding=1)
        pred_x, gt_x = stack_x[: n * c], stack_x[n * c:]
        pred_y, gt_y = stack_y[: n * c], stack_y[n * c:]
        return torch.abs(pred_x - gt_x) + torch.abs(pred_y - gt_y)
