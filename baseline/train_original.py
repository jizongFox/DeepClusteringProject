from pathlib import Path
from pprint import pprint
from typing import Dict, Union, Type, Tuple

from deepclustering.manager import ConfigManger
from deepclustering.model import Model, to_Apex
from torch.utils.data import DataLoader
from deepclustering.utils import fix_all_seed
from deepclustering.augment.pil_augment import Img2Tensor
import sys

sys.path.insert(0, "../")
import trainer

DATA_PATH = Path("../.data")
DATA_PATH.mkdir(exist_ok=True)


def get_dataloader(
        config: Dict[str, Union[float, int, dict, str]]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    We will use config.Config as the input yaml file to select dataset
    config.DataLoader.transforms (naive or strong) to choose data augmentation for GEO
    :param config:
    :return:
    """
    assert config.get("Trainer").get("name").lower() in ("iicgeo", "imsatvat"), \
        f"Trainer name must be in `iicgeo` and `imsatvat`, " \
            f"given {config.get('Trainer', {}).get('name', 'none').lower()}"
    trainer_name = config["Trainer"]["name"].lower()  # based on trainer name, modify the image transform.

    from torchvision.transforms import Compose
    if config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower() == "cifar.yaml":
        from datasets import (
            cifar10_strong_transform as strong_transforms,  # strong transform is for IIC
            Cifar10ClusteringDatasetInterface as DatasetInterface
        )
        if trainer_name == "imsatvat":
            raise RuntimeError("Run IMSATVAT with cifar using `train_original_IMSAT_CIFAR.py`")
        naive_transforms = {"tf1": Compose([Img2Tensor(False, True)]),
                            "tf2": Compose([Img2Tensor(False, True)]),
                            "tf3": Compose([Img2Tensor(False, True)])}  # redefine naive transform for IMSAT
        print("Checkout CIFAR10 dataset with transforms:")
        train_split_partition = ["train", "val"]
        val_split_partition = ["train", "val"]
    elif config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower() == "mnist.yaml":
        from datasets import (
            mnist_strong_transform as strong_transforms,  # strong transform is for IIC
            MNISTClusteringDatasetInterface as DatasetInterface,
        )
        naive_transforms = {"tf1": Compose([Img2Tensor(False, True)]),
                            "tf2": Compose([Img2Tensor(False, True)]),
                            "tf3": Compose([Img2Tensor(False, True)])}  # redefine naive transform for IMSAT
        print("Checkout MNIST dataset with transforms:")
        train_split_partition = ["train", "val"]
        val_split_partition = ["train", "val"]
    else:
        raise NotImplementedError(
            config.get("Config", DEFAULT_CONFIG).split("_")[-1].lower()
        )

    img_transforms = {"imsatvat": naive_transforms, "iicgeo": strong_transforms}.get(trainer_name)

    print("image transformations:")
    pprint(img_transforms)

    train_loader_A = DatasetInterface(
        data_root=DATA_PATH,
        split_partitions=train_split_partition,
        **{k: v for k, v in merged_config["DataLoader"].items() if k != "transforms"}
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
        **{k: v for k, v in merged_config["DataLoader"].items() if k != "transforms"}
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
        **{k: v for k, v in merged_config["DataLoader"].items() if k != "transforms"}
    ).ParallelDataLoader(img_transforms["tf3"])
    return train_loader_A, train_loader_B, val_loader


def get_trainer(
        config: Dict[str, Union[float, int, dict]]
) -> Type[trainer.ClusteringGeneralTrainer]:
    assert config.get("Trainer").get("name"), config.get("Trainer").get("name")
    trainer_mapping: Dict[str, Type[trainer.ClusteringGeneralTrainer]] = {
        "iicgeo": trainer.IICGeoTrainer,  # the basic iic
        "imsatvat": trainer.IMSATVATTrainer,  # imsat with vat
    }
    Trainer = trainer_mapping.get(config.get("Trainer").get("name").lower())
    assert Trainer, config.get("Trainer").get("name")
    return Trainer


if __name__ == '__main__':
    DEFAULT_CONFIG = "config_MNIST.yaml"
    merged_config = ConfigManger(
        DEFAULT_CONFIG_PATH=DEFAULT_CONFIG, verbose=False, integrality_check=True
    ).config
    pprint(merged_config)
    # for reproducibility
    if merged_config.get("Seed"):
        fix_all_seed(merged_config.get("Seed"))

    # get train loaders and validation loader
    train_loader_A, train_loader_B, val_loader = get_dataloader(merged_config)

    # create model:
    model = Model(
        arch_dict=merged_config["Arch"],
        optim_dict=merged_config["Optim"],
        scheduler_dict=merged_config["Scheduler"],
    )
    # if use automatic precision mixture training
    model = to_Apex(model, opt_level=None, verbosity=0)

    # get specific trainer
    Trainer = get_trainer(merged_config)

    clusteringTrainer = Trainer(
        model=model,
        train_loader_A=train_loader_A,
        train_loader_B=train_loader_B,
        val_loader=val_loader,
        config=merged_config,
        **merged_config["Trainer"]
    )
    clusteringTrainer.start_training()
    clusteringTrainer.clean_up(wait_time=3)
