import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def get_params(cfg, size):
    w, h = size
    new_h = h
    new_w = w
    if 'resize' in cfg.image_preprocess:
        new_h = new_w = cfg.load_size
    elif 'scalewidth' in cfg.image_preprocess:
        new_w = cfg.load_size
        new_h = cfg.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - cfg.crop_size))
    y = random.randint(0, np.maximum(0, new_h - cfg.crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(cfg, params, preprocess_mode, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in preprocess_mode:
        osize = [cfg.load_size, cfg.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scalewidth' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, cfg.load_size, method)))

    if 'crop' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], cfg.crop_size)))

    if 'colorjitter' in preprocess_mode:
        transform_list.append(transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8))

    if 'randomgray' in preprocess_mode:
        transform_list.append(transforms.RandomGrayscale(p=0.2))

    if preprocess_mode == 'fixed':
        w = cfg.load_size
        h = round(w * cfg.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if cfg.phase=='train':
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize(cfg.mean, cfg.std)]
    return transforms.Compose(transform_list)


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
