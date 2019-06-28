from functools import reduce
from typing import List, Callable

import PIL
from deepclustering.augment import pil_augment
from torchvision import transforms

from .clustering_helper import ClusterDatasetInterface
from .svhn import SVHN

__all__ = ['SVHNClusteringDatasetInterface', 'default_svhn_img_transform']


class SVHNClusteringDatasetInterface(ClusterDatasetInterface):
    ALLOWED_SPLIT = ['train', 'test']

    def __init__(self, data_root=None, split_partitions: List[str] = [], batch_size: int = 1,
                 shuffle: bool = False,
                 num_workers: int = 1, pin_memory: bool = True) -> None:
        super().__init__(SVHN, data_root, split_partitions, batch_size, shuffle, num_workers, pin_memory)

    def _creat_concatDataset(self, image_transform: Callable, target_transform: Callable, dataset_dict: dict = {}):
        for split in self.split_partitions:
            assert split in self.ALLOWED_SPLIT, f"Allowed split in SVHN:{self.ALLOWED_SPLIT}, given {split}."

        _datasets = []
        for split in self.split_partitions:
            dataset = self.DataClass(self.data_root, split=split,
                                     transform=image_transform, target_transform=target_transform,
                                     download=True, **dataset_dict)
            _datasets.append(dataset)
        serial_dataset = reduce(lambda x, y: x + y, _datasets)
        return serial_dataset


default_svhn_img_transform = {
    "tf1":
        transforms.Compose([
            pil_augment.RandomChoice(transforms=[
                pil_augment.RandomCrop(size=(20, 20), padding=None),
                pil_augment.CenterCrop(size=(20, 20))
            ]),
            pil_augment.Resize(size=24, interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor()
        ]),
    "tf2":
        transforms.Compose([
            pil_augment.RandomApply(transforms=[
                transforms.RandomRotation(
                    degrees=(-25.0, 25.0),
                    resample=False,
                    expand=False
                )], p=0.5),
            pil_augment.RandomChoice(
                transforms=[
                    pil_augment.RandomCrop(size=(16, 16), padding=None),
                    pil_augment.RandomCrop(size=(20, 20), padding=None),
                    pil_augment.RandomCrop(size=(24, 24), padding=None),
                ]),
            pil_augment.Resize(size=24, interpolation=PIL.Image.BILINEAR),
            transforms.ColorJitter(
                brightness=[0.6, 1.4],
                contrast=[0.6, 1.4],
                saturation=[0.6, 1.4],
                hue=[-0.125, 0.125]
            ),
            transforms.ToTensor()
        ]),
    "tf3":
        transforms.Compose([
            pil_augment.CenterCrop(size=(20, 20)),
            pil_augment.Resize(size=24, interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor()
        ]),

}
