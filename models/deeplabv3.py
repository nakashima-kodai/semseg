import os
import torch
import horovod.torch as hvd
import hydra
from torch import nn
from torch.nn import functional as F
from torchvision import models
from models import IntermediateLayerGetter


def convert_sync_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = hvd.SyncBatchNorm(
            module.num_features, module.eps, module.momentum,
            module.affine, module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child))
    del module
    return module_output


class DeepLabv3R50(nn.Module):
    def __init__(self, cfg):
        super(DeepLabv3R50, self).__init__()

        # self.backbone = models.segmentation.deeplabv3_resnet50().backbone
        backbone = models.resnet50(pretrained=cfg.imgnet_pretrained)
        self.backbone = IntermediateLayerGetter(backbone, {'layer4': 'out'})
        self.head = DeepLabHead(2048, cfg.num_classes)

        if cfg.norm_type == 'syncbatch':
            self.backbone = convert_sync_batchnorm(self.backbone)
            self.head = convert_sync_batchnorm(self.head)

        if cfg.phase == 'val' or cfg.phase == 'test':
            save_path = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_pth)
            backbone_weights = {}
            head_weights = {}
            state = torch.load(save_path)['model']
            for k, v in state.items():
                if 'backbone' in k:
                    k = k.replace('backbone.', '')
                    backbone_weights[k] = v
                elif 'head' in k:
                    k = k.replace('head.', '')
                    head_weights[k] = v
            self.backbone.load_state_dict(backbone_weights)
            self.head.load_state_dict(head_weights)
            print('load trained weight from {}'.format(save_path))            

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)['out']
        x = self.head(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
