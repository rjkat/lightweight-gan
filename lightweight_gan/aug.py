import torch
from torch import nn
from random import random

from PIL import Image
from torchvision import transforms
from functools import partial


from lightweight_gan.diff_augment import DiffAugment
from contextlib import contextmanager
from torch.utils.data import Dataset
from pathlib import Path
import torchvision

EXTS = ['jpg', 'jpeg', 'png', 'tiff']

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if alpha is None and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


@contextmanager
def null_context():
    yield

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False, **kwargs):
        context = torch.no_grad if detach else null_context

        with context():
            if random() < prob:
                images = random_hflip(images, prob=0.5)
                images = DiffAugment(images, types=types)

        return self.D(images, **kwargs)

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        transparent = False,
        greyscale = False,
        aug_prob = 0.
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        if transparent:
            num_channels = 4
            pillow_mode = 'RGBA'
            expand_fn = expand_greyscale(transparent)
        elif greyscale:
            num_channels = 1
            pillow_mode = 'L'
            expand_fn = identity()
        else:
            num_channels = 3
            pillow_mode = 'RGB'
            expand_fn = expand_greyscale(transparent)

        convert_image_fn = partial(convert_image_to, pillow_mode)

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_fn)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)