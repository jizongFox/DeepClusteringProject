import sys
from typing import Dict, Union, Type, Tuple

sys.path.insert(0, "../")
from deepclustering.manager import ConfigManger
from deepclustering.model import Model, to_Apex
from deepclustering.utils import Identical, fix_all_seed
from torch.utils.data import DataLoader

# from trainer import ClusteringGeneralTrainer_backup as trainer
import trainer
from baseline.cifarDataset import Cifar10FeatureClusteringInterface

DATA_PATH = "../.data/cifar_features.pth"


def get_dataloader(
        batch_size=10,
        shuffle=False,
        num_workers=4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tf_transforms = {
        "tf1": Identical(),
        "tf2": Identical(),
        "tf3": Identical(),
    }

    cifar10_datahandler = Cifar10FeatureClusteringInterface(
        data_path=str(DATA_PATH), batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    train_loader_A = cifar10_datahandler.ParallelDataLoader(tf_transforms["tf1"], tf_transforms["tf2"])
    train_loader_B = cifar10_datahandler.ParallelDataLoader(tf_transforms["tf1"], tf_transforms["tf2"])
    val_loader = cifar10_datahandler.ParallelDataLoader(tf_transforms["tf3"])

    return train_loader_A, train_loader_B, val_loader


def get_trainer(
        config: Dict[str, Union[float, int, dict]]
) -> Type[trainer.ClusteringGeneralTrainer]:
    assert config.get("Trainer").get("name"), config.get("Trainer").get("name")
    trainer_mapping: Dict[str, Type[trainer.ClusteringGeneralTrainer]] = {
        "imsat": trainer.IMSATAbstractTrainer,  # imsat without any regularization
        "imsatvat": trainer.IMSATVATTrainer,
    }
    Trainer = trainer_mapping.get(config.get("Trainer").get("name").lower())
    assert Trainer, config.get("Trainer").get("name")
    return Trainer


DEFAULT_CONFIG = "config_CIFAR.yaml"

merged_config = ConfigManger(
    DEFAULT_CONFIG_PATH=DEFAULT_CONFIG, verbose=True, integrality_check=True
).config
train_loader_A, train_loader_B, val_loader = get_dataloader(**merged_config["DataLoader"])

# create model:
model = Model(
    arch_dict=merged_config["Arch"],
    optim_dict=merged_config["Optim"],
    scheduler_dict=merged_config["Scheduler"],
)
model = to_Apex(model, opt_level=None, verbosity=0)

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
clusteringTrainer.clean_up()
