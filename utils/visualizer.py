import os
import numpy as np
import torch
import torchvision
import wandb
from PIL import Image


class Visualizer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = wandb.init(
            name=cfg.name, project='semseg', config=cfg, dir=os.getcwd(), group=cfg.group, id=cfg.id
        )

    def save_architecture(self, model):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()

        save_path = os.path.join(wandb.run.dir, self.cfg.model_name + '.txt')
        with open(save_path, 'wt') as f:
            f.write(str(model))
            f.write('\nTotal number of parameters : %.3f M\n' % (num_params / 1e6))

    def log_losses(self, log_items, global_steps):
        for k, v in log_items.items():
            self.logger.log({k: v}, step=global_steps)
    
    def log_visuals(self, visuals, global_steps):
        for k, v in visuals.items():
            v = torchvision.utils.make_grid(v, nrow=8, padding=1)
            if 'label' in k:
                v = Colorizer()(v)
            self.logger.log({k: wandb.Image(v)}, step=global_steps)


class Colorizer():
    def __init__(self):
        self.cmap = np.array([
            (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
            (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
            (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), 
            (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0)
        ], dtype=np.float32)
        self.cmap = torch.from_numpy(self.cmap)

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.FloatTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image