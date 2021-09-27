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
from PIL import Image as PImage

class RescaleSQR(object):
    def __init__(self, output_size, fill_color=(0, 0, 0)):
        self.fill_color = fill_color
        self.output_size = output_size

    def __make_square(self, im, min_size=256, fill_color=(0, 0, 0)):
        im_x, im_y = im.size

        largest_side = max(im_x, im_y)
        scale_factor = min_size / largest_side
        scaled_x, scaled_y = (int(im_x * scale_factor),
                              int(im_y * scale_factor))
        im = im.resize((scaled_x, scaled_y))

        size = max(min_size, scaled_x, scaled_y)
        new_im = PImage.new('RGB', (size, size), fill_color)
        new_im.paste(im, (int((size - scaled_x) / 2),
                          int((size - scaled_y) / 2)))
        return new_im

    def __call__(self, sample):
        return self.__make_square(sample, self.output_size, self.fill_color)

class RandomBottomFlip(object):
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, sample):
        (size_x, size_y) = sample.size
        new_im = PImage.new('RGB', (size_x, size_y))

        top_side_rect = (0,0, size_x, int(size_y / 2))
        top_side = sample.crop(top_side_rect)
        new_im.paste(top_side, top_side_rect)

        bottom_side_rect = (0, int(size_y / 2), size_x, size_y)
        bottom_side = sample.crop(bottom_side_rect)

        r = random.uniform(0, 1)
        if r <= self.p:
            bottom_side = bottom_side.transpose(PImage.FLIP_LEFT_RIGHT)
        new_im.paste(bottom_side, bottom_side_rect)

        return new_im

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
            RescaleSQR(size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_id == "transform_flip":
        return Compose([
            RescaleSQR(size),
            RandomHorizontalFlip(p=1.0),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_id == "transform_hue":
        return Compose([
            RescaleSQR(size),
            ColorJitter(hue=0.5),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_id == "transform_hue_flip":
        return Compose([
            RescaleSQR(size),
            ColorJitter(hue=0.5),
            RandomHorizontalFlip(p=1.0),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_id == "transform_ppile":
        return Compose([
            RescaleSQR(size),
            RandomRotation(30, fill=(255,255,255)),
            RandomAffine(0, translate=(5/size, 5/size),
                         fill=(255,255,255)),
            ColorJitter(hue=0.5),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])
    elif transform_id == "transform_ppile_aggressive":
        return Compose([
            RescaleSQR(size),
            RandomRotation(30, fill=(255,255,255)),
            RandomAffine(0, translate=(5/size, 5/size),
                         fill=(255,255,255)),
            ColorJitter(hue=0.5),
            RandomBottomFlip(p=0.75),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])