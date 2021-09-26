"""
-------------------------------------------------
   File Name:    transforms.py
   Author:       Zhonghao Huang
   Date:         2019/10/22
   Description:
-------------------------------------------------
"""

from torchvision.transforms import ToTensor, Normalize, Compose, Resize, RandomHorizontalFlip
from torchvision.transforms import CenterCrop, ColorJitter, RandomRotation, RandomAffine

def get_transform(new_size=None):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """


    if new_size is not None:
        image_transform = Compose([
            RandomHorizontalFlip(),
            Resize(new_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    else:
        image_transform = Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    return image_transform

def get_transform_with_id(transform_id, size):
    if transform_id == "transform":
        return Compose([
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_id == "transform_flip":
        return Compose([
            Resize(IMAGE_SIZE),
            CenterCrop(IMAGE_SIZE),
            RandomHorizontalFlip(p=1.0),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_id == "transform_hue":
        return Compose([
            Resize(IMAGE_SIZE),
            CenterCrop(IMAGE_SIZE),
            ColorJitter(hue=0.5),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_id == "transform_hue_flip":
        return Compose([
            Resize(IMAGE_SIZE),
            CenterCrop(IMAGE_SIZE),
            ColorJitter(hue=0.5),
            RandomHorizontalFlip(p=1.0),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_id == "transform_ppile":
        return Compose([
            Resize(IMAGE_SIZE),
            CenterCrop(IMAGE_SIZE),
            RandomRotation(30, fill=(255,255,255)),
            RandomAffine(0, translate=(5/IMAGE_SIZE, 5/IMAGE_SIZE),
                         fill=(255,255,255)),
            ColorJitter(hue=0.5),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])