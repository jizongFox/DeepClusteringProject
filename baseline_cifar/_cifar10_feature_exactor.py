import sys

import torch

sys.path.insert(0, "../")
import numpy as np
from torchvision.models import resnet50
from deepclustering.utils import Identical, tqdm
from datasets import Cifar10ClusteringDatasetInterface
from torchvision import transforms
import os

network = resnet50(pretrained=True)
network.fc = Identical()
network.eval()
network.cuda()
img = torch.randn(10, 3, 224, 224)

dataroot = "../.data"

cifar_handler = Cifar10ClusteringDatasetInterface(
    data_root=dataroot,
    split_partitions=["train", "val"],
    batch_size=30,
    shuffle=False,
    pin_memory=True)

cifar_loader = cifar_handler.ParallelDataLoader(
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
)
features = []
results = []
with torch.no_grad():
    for i, data in enumerate(tqdm(cifar_loader)):
        feats = network(data[0][0].cuda())
        features.append(feats.cpu().numpy())
        results.append(data[0][1].numpy())
numpy_features = np.concatenate(features, axis=0)
numpy_targets = np.concatenate(results, axis=0)

torch.save({"features": numpy_features,
            "targets": numpy_targets}, os.path.join(dataroot, "cifar_features.pth"))
