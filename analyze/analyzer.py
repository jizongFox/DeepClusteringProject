from colorsys import hsv_to_rgb
from copy import deepcopy as dcp
from typing import Tuple, Dict

import numpy as np
import torch
from PIL import Image
from deepclustering import ModelMode
from deepclustering.meters import MeterInterface, ConfusionMatrix, AverageValueMeter
from deepclustering.model import Model
from deepclustering.utils import tqdm_, nice_dict
from deepclustering.utils.classification.assignment_mapping import hungarian_match, flat_acc
from deepclustering.writer import DrawCSV2
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from trainer import ClusteringGeneralTrainer


class LinearNet(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, input):
        return self.fc(input)


class AnalyzeInference(ClusteringGeneralTrainer):
    checkpoint_identifier = "best.pth"

    def __init__(self, model: Model, train_loader_A: DataLoader, train_loader_B: DataLoader, val_loader: DataLoader,
                 criterion: nn.Module = nn.CrossEntropyLoss(), max_epoch: int = 100,
                 save_dir: str = "AnalyzerTrainer",
                 checkpoint_path: str = None, device="cpu", head_control_params: Dict[str, int] = {"B": 1},
                 use_sobel: bool = False, config: dict = None, **kwargs) -> None:
        super().__init__(model, train_loader_A, train_loader_B, val_loader, criterion, max_epoch, save_dir,
                         checkpoint_path, device, head_control_params, use_sobel, config, **kwargs)

        assert self.checkpoint, "checkpoint must be provided in `AnalyzeInference`."

    # for 10 point projection
    def save_plot(self) -> None:
        """
        using IIC method to show the evaluation, only for MNIST dataset
        """
        assert self.val_loader.dataset_name == "mnist", \
            "`save tsne plot` is only implemented for MNIST dataset, given {self.val_loader.dataset_name}."

        def get_coord(probs, num_classes):
            # computes coordinate for 1 sample based on probability distribution over c
            coords_total = np.zeros(2, dtype=np.float32)
            probs_sum = probs.sum()

            fst_angle = 0.

            for c in range(num_classes):
                # compute x, y coordinates
                coords = np.ones(2) * 2 * np.pi * (float(c) / num_classes) + fst_angle
                coords[0] = np.sin(coords[0])
                coords[1] = np.cos(coords[1])
                coords_total += (probs[c] / probs_sum) * coords
            return coords_total

        GT_TO_ORDER = [2, 5, 3, 8, 6, 7, 0, 9, 1, 4]
        with torch.no_grad():
            best_score, (target, soft_preds) = self._eval_loop(val_loader=self.val_loader, epoch=100000,
                                                               mode=ModelMode.EVAL,
                                                               return_soft_predict=True)
        print(f"best score: {best_score}")
        soft_preds = soft_preds.numpy()
        average_images = self.plot_cluster_average_images(self.val_loader, soft_preds)

        # render point cloud in GT order ---------------------------------------------
        hues = torch.linspace(0.0, 1.0, self.model.arch_dict["output_k_B"] + 1)[0:-1]  # ignore last one
        best_colours = [list((np.array(hsv_to_rgb(hue, 0.8, 0.8)) * 255.).astype(
            np.uint8)) for hue in hues]

        all_colours = [best_colours]

        for colour_i, colours in enumerate(all_colours):
            scale = 50  # [-1, 1] -> [-scale, scale]
            border = 24  # averages are in the borders
            point_half_side = 1  # size 2 * pixel_half_side + 1

            half_border = int(border * 0.5)

            image = np.ones((2 * (scale + border), 2 * (scale + border), 3),
                            dtype=np.uint8) * 255

            for i in range(len(soft_preds)):
                # in range [-1, 1] -> [0, 2 * scale] -> [border, 2 * scale + border]
                coord = get_coord(soft_preds[i, :], num_classes=self.model.arch_dict["output_k_B"])
                coord = (coord * 0.75 * scale + scale).astype(np.int32)
                coord += border
                pt_start = coord - point_half_side
                pt_end = coord + point_half_side

                render_c = GT_TO_ORDER[target[i]]
                colour = (np.array(colours[render_c])).astype(np.uint8)
                image[pt_start[0]:pt_end[0], pt_start[1]:pt_end[1], :] = np.reshape(
                    colour, (1, 1, 3))
            # add average images
            for i in range(10):
                pred = np.zeros(10)
                pred[i] = 1
                coord = get_coord(pred, 10)
                coord = (coord * 1.2 * scale + scale).astype(np.int32)
                coord += border
                pt_start = coord - half_border
                pt_end = coord + half_border
                image[pt_start[0]:pt_end[0], pt_start[1]:pt_end[1], :] = average_images[GT_TO_ORDER[i]].unsqueeze(
                    2).repeat(
                    [1, 1, 3]) * 255.0

            # save to out_dir ---------------------------
            img = Image.fromarray(image)
            img.save(self.save_dir / f"best_tsne_{colour_i}.png")

    @staticmethod
    def plot_cluster_average_images(val_loader, soft_pred):
        assert val_loader.dataset_name == "mnist", \
            f"save tsne plot is only implemented for MNIST dataset, given {val_loader.dataset_name}."

        average_images = [torch.zeros(24, 24) for _ in range(10)]

        counter = 0
        for image_labels in tqdm_(val_loader):
            images, gt, *_ = list(zip(*image_labels))
            # only take the tf3 image and gts, put them to self.device
            images, gt = images[0], gt[0]
            for i, img in enumerate(images):
                average_images[soft_pred[counter + i].argmax()] += img.squeeze() * soft_pred[counter + i].max()

            counter += len(images)
        assert counter == val_loader.dataset.__len__()
        average_images = [average_image / (counter / 10) for average_image in average_images]
        return average_images

    # 10 point projection ends

    # for tsne projection
    def draw_tsne(self, val_loader, num_samples=1000):
        self.model.eval()
        assert val_loader.dataset_name == "mnist", \
            f"save tsne plot is only implemented for MNIST dataset, given {self.val_loader.dataset_name}."
        from deepclustering.arch.classification.IIC.net6c_two_head import ClusterNet6cTwoHead
        assert isinstance(self.model.torchnet,
                          ClusterNet6cTwoHead), f"self.model must be ClusterNet6cTwoHead, given {self.model}"

        images, features, targets = self.feature_exactor(conv_name="trunk", val_loader=self.val_loader)
        idx = torch.randperm(targets.size(0))[:num_samples]
        self.writer.add_embedding(mat=features[idx], metadata=targets[idx], label_img=images[idx],
                                  global_step=10000)

    # feature extraction
    def feature_exactor(self, conv_name: str = "trunk", val_loader: DataLoader = None) -> Tuple[Tensor, Tensor, Tensor]:
        assert isinstance(val_loader, DataLoader)
        _images = []
        _features = []
        _targets = []
        _preds = []

        def hook(module, input, output):
            _features.append(output.cpu().detach())

        exec(f"handler=self.model.torchnet.{conv_name}.register_forward_hook(hook)")
        for batch, image_labels in enumerate(val_loader):
            img, gt, *_ = list(zip(*image_labels))
            # only take the tf3 image and gts, put them to self.device
            img, gt = img[0].to(self.device), gt[0].to(self.device)
            # if use sobel filter
            if self.use_sobel:
                img = self.sobel(img)
            # using default head_B for inference, _pred should be a list of simplex by default.
            _pred = self.model.torchnet(img, head="B")[0]
            _images.append(img.cpu())
            _targets.append(gt.cpu())
            _preds.append(_pred.max(1)[1].cpu())
        features = torch.cat(_features, 0)
        targets = torch.cat(_targets, 0)
        images = torch.cat(_images, 0)
        preds = torch.cat(_preds, 0)
        remaped_pred, _ = hungarian_match(
            flat_preds=preds,
            flat_targets=targets,
            preds_k=self.model.arch_dict["output_k_B"],
            targets_k=self.model.arch_dict["output_k_B"]
        )
        acc = flat_acc(remaped_pred, targets)
        assert features.shape[0] == targets.shape[0]
        exec("handler.remove()")
        print(f"Feature exaction ends with acc: {acc:.4f}")
        return images, features, targets

    def linear_retraining(self, conv_name: str, lr = 1e-3):
        """
        Calling point to execute retraining
        :param conv_name:
        :return:
        """
        print(f"conv_name: {conv_name}, feature extracting..")

        def _linear_train_loop(train_loader, epoch):
            train_loader_ = tqdm_(train_loader)
            for batch_num, (feature, gt) in enumerate(train_loader_):
                feature, gt = feature.to(self.device), gt.to(self.device)
                pred = linearnet(feature)
                loss = self.criterion(pred, gt)
                linearOptim.zero_grad()
                loss.backward()
                linearOptim.step()
                linear_meters["train_loss"].add(loss.item())
                linear_meters["train_acc"].add(pred.max(1)[1], gt)
                report_dict = {
                    "tra_acc": linear_meters["train_acc"].summary()["acc"],
                    "loss": linear_meters["train_loss"].summary()["mean"],
                }
                train_loader_.set_postfix(report_dict)

            print(f"  Training epoch {epoch}: {nice_dict(report_dict)} ")

        def _linear_eval_loop(val_loader, epoch) -> Tensor:
            val_loader_ = tqdm_(val_loader)
            for batch_num, (feature, gt) in enumerate(val_loader_):
                feature, gt = feature.to(self.device), gt.to(self.device)
                pred = linearnet(feature)
                linear_meters["val_acc"].add(pred.max(1)[1], gt)
                report_dict = {"val_acc": linear_meters["train_acc"].summary()["acc"]}
                val_loader_.set_postfix(report_dict)
            print(f"Validating epoch {epoch}: {nice_dict(report_dict)} ")
            return linear_meters["val_acc"].summary()["acc"]

        # building training and validation set based on extracted features
        train_loader = dcp(self.val_loader)
        train_loader.dataset.datasets = (train_loader.dataset.datasets[0].datasets[0],)
        val_loader = dcp(self.val_loader)
        val_loader.dataset.datasets = (val_loader.dataset.datasets[0].datasets[1],)
        _, train_features, train_targets = self.feature_exactor(conv_name, train_loader)
        print(f"training_feature_shape: {train_features.shape}")
        train_features = train_features.view(train_features.size(0), -1)
        _, val_features, val_targets = self.feature_exactor(conv_name, val_loader)
        val_features = val_features.view(val_features.size(0), -1)
        print(f"val_feature_shape: {val_features.shape}")

        train_dataset = TensorDataset(train_features, train_targets)
        val_dataset = TensorDataset(val_features, val_targets)
        Train_DataLoader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        Val_DataLoader = DataLoader(val_dataset, batch_size=100, shuffle=False)

        # network and optimization
        linearnet = LinearNet(num_features=train_features.size(1), num_classes=self.model.arch_dict["output_k_B"])
        linearOptim = torch.optim.Adam(linearnet.parameters(), lr=lr)
        linearnet.to(self.device)

        # meters
        meter_config = {
            "train_loss": AverageValueMeter(),
            "train_acc": ConfusionMatrix(self.model.arch_dict["output_k_B"]),
            "val_acc": ConfusionMatrix(self.model.arch_dict["output_k_B"])
        }
        linear_meters = MeterInterface(meter_config)
        drawer = DrawCSV2(save_dir=self.save_dir, save_name=f"retraining_from_{conv_name}.png",
                          columns_to_draw=["train_loss_mean",
                                           "train_acc_acc",
                                           "val_acc_acc"])
        for epoch in range(self.max_epoch):
            _linear_train_loop(Train_DataLoader, epoch)
            _ = _linear_eval_loop(Val_DataLoader, epoch)
            linear_meters.step()
            linear_meters.summary().to_csv(self.save_dir / f"retraining_from_{conv_name}.csv")
            drawer.draw(linear_meters.summary())
