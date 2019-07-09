__all__ = ["Cifar10_feature", "Cifar10FeatureClusteringInterface"]

from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch
import torch.utils.data as data
from deepclustering.dataset.classification.clustering_helper import ClusterDatasetInterface


class Cifar10_feature(data.Dataset):
    """
    Feature extracted version of the cifar dataset, dealed with numpy features of cifar10 dataset.
    The numpy features are generated using resnet50 network, before which the cifar 10 images are
    resized to 224 \times 224 and normalized with ImageNet mean and std.
    Training and validation datasets are mixed together so that this dataset can only be applied to clustering
    problems.
    """

    def __init__(self, data_path: str = None,
                 img_transform: Callable[[np.ndarray], torch.Tensor] = None,
                 target_transform: Callable[[np.ndarray], Any] = None) -> None:
        """
        :param data_path: cifar10_feature.pth path
        :type data_path: str
        """
        super().__init__()
        assert Path(data_path).exists() and Path(data_path).is_file()
        self.data_path: str = data_path
        self.img_transform = img_transform
        self.target_transform = target_transform

        cifar10_features = torch.load(self.data_path)
        self.imgs: np.ndarray = cifar10_features["features"]
        self.targets: np.ndarray = cifar10_features["targets"]
        del cifar10_features

        assert self.imgs.shape == (60000, 2048)
        assert self.targets.shape == (60000,)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        img, target = self.imgs[index], self.targets[index]
        if self.img_transform:
            img = self.img_transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target


class Cifar10FeatureClusteringInterface(ClusterDatasetInterface):
    def __init__(self, data_path: str, batch_size: int = 1, shuffle: bool = False, num_workers: int = 1,
                 pin_memory: bool = True,
                 drop_last=False) -> None:
        super().__init__(Cifar10_feature, data_path, [""], batch_size, shuffle, num_workers, pin_memory,
                         drop_last)

    def _creat_concatDataset(self, image_transform: Callable, target_transform: Callable, dataset_dict: dict = {}):
        return self.DataClass(self.data_root, img_transform=image_transform, target_transform=target_transform)
