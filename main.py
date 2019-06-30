from pathlib import Path
from typing import Dict, Union, Type, Tuple

from deepclustering.manager import ConfigManger
from deepclustering.model import Model, to_Apex
from deepclustering.utils import fix_all_seed
from torch.utils.data import DataLoader

import ClusteringGeneralTrainer as trainer

DATA_PATH = Path(".data")
DATA_PATH.mkdir(exist_ok=True)

fix_all_seed(2)


def get_dataloader(
        config: Dict[str, Union[float, int, dict, str]]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    get datasets for IIC project
    :param config:
    :return:
    """
    if config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower() == "cifar.yaml":
        from datasets import (
            default_cifar10_img_transform as img_transforms,
            Cifar10ClusteringDatasetInterface as DatasetInterface,
        )

        train_split_partition = ["train", "val"]
        val_split_partition = ["train", "val"]

    elif config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower() == "mnist.yaml":
        from datasets import (
            default_mnist_strong_transform as img_transforms,
            MNISTClusteringDatasetInterface as DatasetInterface,
        )
        train_split_partition = ["train", "val"]
        val_split_partition = ["train", "val"]
    elif config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower() == "stl10.yaml":
        from datasets import (
            default_stl10_img_transform as img_transforms,
            STL10ClusteringDatasetInterface as DatasetInterface,
        )
        train_split_partition = ["train", "test", "train+unlabeled"]
        val_split_partition = ["train", "test"]
    elif config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower() == "svhn.yaml":
        from datasets import (
            default_svhn_img_transform as img_transforms,
            SVHNClusteringDatasetInterface as DatasetInterface,
        )
        train_split_partition = ["train", "test"]
        val_split_partition = ["train", "test"]
    else:
        raise NotImplementedError(
            config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower()
        )
    train_loader_A = DatasetInterface(
        data_root=DATA_PATH,
        split_partitions=train_split_partition,
        **merged_config["DataLoader"]
    ).ParallelDataLoader(
        img_transforms["tf1"],
        img_transforms["tf2"],
        img_transforms["tf2"],
        img_transforms["tf2"],
        img_transforms["tf2"],
        img_transforms["tf2"],
    )
    train_loader_B = DatasetInterface(
        data_root=DATA_PATH,
        split_partitions=train_split_partition,
        **merged_config["DataLoader"]
    ).ParallelDataLoader(
        img_transforms["tf1"],
        img_transforms["tf2"],
        img_transforms["tf2"],
        img_transforms["tf2"],
        img_transforms["tf2"],
        img_transforms["tf2"],
    )
    val_loader = DatasetInterface(
        data_root=DATA_PATH,
        split_partitions=val_split_partition,
        **merged_config["DataLoader"]
    ).ParallelDataLoader(img_transforms["tf3"])
    return train_loader_A, train_loader_B, val_loader


def get_trainer(
        config: Dict[str, Union[float, int, dict]]
) -> Type[trainer.ClusteringGeneralTrainer]:
    assert config.get("Trainer").get("name"), config.get("Trainer").get("name")
    trainer_mapping: Dict[str, Type[trainer.ClusteringGeneralTrainer]] = {
        "iicgeo": trainer.IICGeoTrainer,  # the basic iic
        "iicmixup": trainer.IICMixupTrainer,  # the basic IIC with mixup as the data augmentation
        "iicvat": trainer.IICVATTrainer,  # the basic iic with VAT as the basic data augmentation
        "iicgeovat": trainer.IICGeoVATTrainer,  # IIC with geo and vat as the data augmentation
        "imsat": trainer.IMSATAbstractTrainer,  # imsat without any regularization
        "imsatvat": trainer.IMSATVATTrainer,
        "imsatmixup": trainer.IMSATMixupTrainer,
        "imsatvatgeo": trainer.IMSATVATGeoTrainer,
        "imsatvatgeomixup": trainer.IMSATVATGeoMixupTrainer,
    }
    Trainer = trainer_mapping.get(config.get("Trainer").get("name").lower())
    assert Trainer, config.get("Trainer").get("name")
    return Trainer


DEFAULT_CONFIG = "config/config_MNIST.yaml"

merged_config = ConfigManger(
    DEFAULT_CONFIG_PATH=DEFAULT_CONFIG, verbose=True, integrality_check=True
).config
train_loader_A, train_loader_B, val_loader = get_dataloader(merged_config)
# tf1 = Compose([ToTensor()])
# tf2 = Compose([RandomCrop((28, 28), padding=2), ToTensor()])
# train_loader_A = MNISTClusteringDatasetInterface(
#     data_root=DATA_PATH,
#     split_partitions=["train", "val"],
#     **merged_config["DataLoader"]
# ).ParallelDataLoader(
#     tf1,
#     tf2,
#     tf2,
#     tf2,
#     tf2
# )
# train_loader_B = MNISTClusteringDatasetInterface(
#     data_root=DATA_PATH,
#     split_partitions=["train", "val"],
#     **merged_config["DataLoader"]
# ).ParallelDataLoader(
#     tf1,
#     tf2,
#     tf2,
#     tf2,
#     tf2
# )
#
# val_loader = MNISTClusteringDatasetInterface(
#     data_root=DATA_PATH,
#     split_partitions=["train", "val"],
#     **merged_config["DataLoader"]
# ).ParallelDataLoader(tf1)

# create model:
model = Model(
    arch_dict=merged_config["Arch"],
    optim_dict=merged_config["Optim"],
    scheduler_dict=merged_config["Scheduler"],
)
model = to_Apex(model, opt_level=None, verbosity=0)

Trainer = get_trainer(merged_config)

trainer = Trainer(
    model=model,
    train_loader_A=train_loader_A,
    train_loader_B=train_loader_B,
    val_loader=val_loader,
    config=merged_config,
    **merged_config["Trainer"]
)
trainer.start_training()
trainer.clean_up()
