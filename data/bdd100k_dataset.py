import os
import torch.utils.data as data
from PIL import Image
from data.transforms import get_params, get_transform
from data.image_folder import make_dataset


class BDD100KDataset(data.Dataset):
    def __init__(self, cfg):
        super(BDD100KDataset, self).__init__()
        self.cfg = cfg

        image_dir = os.path.join(cfg.dataroot, 'images', cfg.phase)
        self.image_paths = sorted(make_dataset(image_dir))

        label_dir = os.path.join(cfg.dataroot, 'labels', cfg.phase)
        self.label_paths = sorted(make_dataset(label_dir))

    def __getitem__(self, index):
        # load image
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        params = get_params(self.cfg, image.size)
        transform_image = get_transform(self.cfg, params, self.cfg.image_preprocess)
        image_tensor = transform_image(image)

        # load label
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')
        transform_label = get_transform(self.cfg, params, self.cfg.label_preprocess, Image.NEAREST, False)
        label_tensor = transform_label(label) * 255
        label_tensor[label_tensor == 255] = self.cfg.num_classes  # ignore label is cfg.num_classes

        input_dict = {'image': image_tensor, 'label': label_tensor}
        return input_dict

    def __len__(self):
        return len(self.image_paths)
