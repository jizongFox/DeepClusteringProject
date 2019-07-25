import os
from pathlib import Path
from shutil import copyfile

from deepclustering.manager import ConfigManger, yaml_load
from deepclustering.model import Model, to_Apex
from deepclustering.utils import fix_all_seed

from main import get_dataloader

DATA_PATH = Path(".data")
from analyze import AnalyzeInference


def get_config(checkpoint_path):
    DEFAULT_CONFIG = None
    assert "mnist" in checkpoint_path.lower() or "cifar" in checkpoint_path.lower() or "svhn" in checkpoint_path.lower()
    if "mnist" in checkpoint_path.lower():
        DEFAULT_CONFIG = os.path.join(checkpoint_path, "config_MNIST.yaml")
    if "cifar" in checkpoint_path.lower():
        DEFAULT_CONFIG = os.path.join(checkpoint_path, "config_CIFAR.yaml")
    if "svhn" in checkpoint_path.lower():
        DEFAULT_CONFIG = os.path.join(checkpoint_path, "config_SVHN.yaml")
    assert DEFAULT_CONFIG
    assert Path(checkpoint_path, "config.yaml").exists()
    copyfile(os.path.join(checkpoint_path, "config.yaml"), DEFAULT_CONFIG)
    assert Path(DEFAULT_CONFIG).exists() and Path(DEFAULT_CONFIG).is_file()
    config = yaml_load(DEFAULT_CONFIG)
    return config, DEFAULT_CONFIG


if __name__ == '__main__':
    # interface: python analyze_main checkpoint_path=runs/07_15_benchmark/strong/cifar_2/iicgeo
    DEFAULT_CONFIG = None
    checkpoint_path = ConfigManger(DEFAULT_CONFIG_PATH=DEFAULT_CONFIG, verbose=True, integrality_check=True).config[
        "checkpoint_path"]
    merged_config, DEFAULT_CONFIG = get_config(checkpoint_path)
    # correct save_dir and checkpoint_dir
    merged_config["Trainer"]["save_dir"] = checkpoint_path
    merged_config["Trainer"]["checkpoint_path"] = checkpoint_path
    merged_config["Trainer"]["max_epoch"] = 300

    # for reproducibility
    fix_all_seed(merged_config.get("Seed", 0))

    # get train loaders and validation loader
    train_loader_A, train_loader_B, val_loader = get_dataloader(merged_config, DEFAULT_CONFIG)

    # create model:
    model = Model(
        arch_dict=merged_config["Arch"],
        optim_dict=merged_config["Optim"],
        scheduler_dict=merged_config["Scheduler"],
    )
    # if use automatic precision mixture training
    model = to_Apex(model, opt_level=None, verbosity=0)
    AnalyzeInference.RUN_PATH = ""

    clusteringTrainer = AnalyzeInference(
        model=model,
        train_loader_A=train_loader_A,
        train_loader_B=train_loader_B,
        val_loader=val_loader,
        config=merged_config,
        **merged_config["Trainer"]
    )
    """ for feature extraction and retrainning
    if "mnist" in DEFAULT_CONFIG.lower() or "svhn" in DEFAULT_CONFIG.lower():
        # for mnist (VGG styple network): `trunk`, `trunk.features[N]`, etc.
        clusteringTrainer.linear_retraining("head_B.heads[0]", lr=1e-4)
        clusteringTrainer.linear_retraining("trunk", lr=1e-4)
        clusteringTrainer.linear_retraining("trunk.features[11]", lr=1e-4)
        clusteringTrainer.linear_retraining("trunk.features[7]", lr=1e-4)
        clusteringTrainer.linear_retraining("trunk.features[3]", lr=1e-4)


    elif "cifar" in DEFAULT_CONFIG.lower():
        # for cifar (resnet style network):
        # `head_B.heads[0] with feature size 10`,
        # `trunk.avgpool with feature size 512`,
        # `trunk.layer4 with feature size 512, 3, 3`, etc...
        for i in range(5):
            try:
                clusteringTrainer.linear_retraining(f"head_B.heads[{i}]", lr=1e-4)
            except Exception as e:
                continue

        clusteringTrainer.linear_retraining("trunk", lr=1e-4)
        clusteringTrainer.linear_retraining("trunk.layer4", lr=1e-4)
        clusteringTrainer.linear_retraining("trunk.layer3", lr=1e-4)
        clusteringTrainer.linear_retraining("trunk.layer2", lr=1e-4)
    else:
        raise NotImplementedError("Only support mnist, cifar, and svhn.")
    """
    clusteringTrainer.supervised_training(use_pretrain=True)
    clusteringTrainer.supervised_training(use_pretrain=False)
