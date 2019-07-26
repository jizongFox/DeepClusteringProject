from torch import nn
import torch


class Temporature(nn.Module):
    def __init__(self, temporature=1.0):
        super().__init__()
        self.T = float(temporature)

    def forward(self, x):
        return x / self.T

    def __repr__(self):
        return f"Temporature={self.T}"


class Image_Pool:

    def __init__(self, image_per_class=20, total_classes=10) -> None:
        super().__init__()
        self.num_per_class = image_per_class
        self.total_classes = total_classes
        self.image_dict = {}
        for i in range(total_classes):
            self.image_dict[i] = []

    def add(self, imgs, gts):
        assert imgs.size(0) == gts.size(0)
        for i, (img, gt) in enumerate(zip(imgs, gts)):
            if len(self.image_dict[gt.item()]) < self.num_per_class:
                self.image_dict[gt.item()].append(img)
            else:
                continue

    def image_pool(self):
        return self.image_dict
