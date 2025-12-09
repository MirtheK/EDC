from models.resnet import resnet10, resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from models.resnet_decoder import resnet10_decoder, resnet18_decoder, resnet50_decoder, wide_resnet50_decoder, resnet34_decoder, resnext50_32x4d_decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import math


def zero_side(p, side=1):
    p[:, :, :side, :] = 0
    p[:, :, :, :side] = 0

    p[:, :, -side:, :] = 0
    p[:, :, :, -side:] = 0

    return p


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class R50_R50(nn.Module):
    def __init__(self,
                 img_size=256,
                 train_encoder=True,
                 stop_grad=True,
                 reshape=True,
                 bn_pretrain=False,
                 anomap_layer=[1, 2, 3]
                 ):
        super().__init__()
        self.edc_encoder = resnet50(pretrained=True)
        self.edc_decoder = resnet50_decoder(pretrained=False, inplanes=[512])
        self.train_encoder = train_encoder
        self.stop_grad = stop_grad
        self.reshape = reshape
        self.bn_pretrain = bn_pretrain
        self.anomap_layer = anomap_layer

    def forward(self, x, ):
        if not self.train_encoder and self.edc_encoder.training:
            self.edc_encoder.eval()
        if self.bn_pretrain and self.edc_encoder.training:
            self.edc_encoder.eval()

        B = x.shape[0]

        e1, e2, e3, e4 = self.edc_encoder(x)
        if not self.train_encoder:
            e4 = e4.detach()
        d1, d2, d3 = self.edc_decoder(e4)

        if (not self.train_encoder) or self.stop_grad:
            e1 = e1.detach()
            e2 = e2.detach()
            e3 = e3.detach()

        # if self.reshape:
        #     l1 = 1. - torch.cosine_similarity(d1.reshape(B, -1), e1.reshape(B, -1), dim=1).mean()
        #     l2 = 1. - torch.cosine_similarity(d2.reshape(B, -1), e2.reshape(B, -1), dim=1).mean()
        #     l3 = 1. - torch.cosine_similarity(d3.reshape(B, -1), e3.reshape(B, -1), dim=1).mean()
        # else:
        #     l1 = 1. - torch.cosine_similarity(d1, e1, dim=1).mean()
        #     l2 = 1. - torch.cosine_similarity(d2, e2, dim=1).mean()
        #     l3 = 1. - torch.cosine_similarity(d3, e3, dim=1).mean()

        # with torch.no_grad():
        #     p1 = 1. - torch.cosine_similarity(d1, e1, dim=1).unsqueeze(1)
        #     p2 = 1. - torch.cosine_similarity(d2, e2, dim=1).unsqueeze(1)
        #     p3 = 1. - torch.cosine_similarity(d3, e3, dim=1).unsqueeze(1)

        # MSE loss instead
        criterion = nn.MSELoss(reduction='none')

        if self.reshape:
            l1 = criterion(d1.reshape(B, -1), e1.reshape(B, -1)).mean()
            l2 = criterion(d2.reshape(B, -1), e2.reshape(B, -1)).mean()
            l3 = criterion(d3.reshape(B, -1), e3.reshape(B, -1)).mean()
        else:
            l1 = criterion(d1, e1).mean()
            l2 = criterion(d2, e2).mean()
            l3 = criterion(d3, e3).mean()

        with torch.no_grad():
            p1 = criterion(d1, e1) 
            p2 = criterion(d2, e2)
            p3 = criterion(d3, e3)
        loss = l1 + l2 + l3

        p2 = F.interpolate(p2, scale_factor=2, mode='trilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=4, mode='trilinear', align_corners=False)

        p_all = [[p1, p2, p3][l - 1] for l in self.anomap_layer]
        p_all = torch.cat(p_all, dim=1).mean(dim=1, keepdim=True)

        with torch.no_grad():
            e1_std = F.normalize(e1.permute(1, 0, 2, 3, 4).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(e2.permute(1, 0, 2, 3, 4).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(e3.permute(1, 0, 2, 3, 4).flatten(1), dim=0).std(dim=1).mean()

        return {'loss': loss, 'p_all': p_all, 'p1': p1, 'p2': p2, 'p3': p3,
                'e1_std': e1_std, 'e2_std': e2_std, 'e3_std': e3_std}

