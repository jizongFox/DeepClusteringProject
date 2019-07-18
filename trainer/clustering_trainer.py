"""
This is the trainer general clustering trainer
"""
import time
from collections import OrderedDict
from colorsys import hsv_to_rgb
from pathlib import Path
from typing import List, Union, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from deepclustering import ModelMode
from deepclustering.augment.pil_augment import SobelProcess
from deepclustering.loss import KL_div
from deepclustering.meters import AverageValueMeter, MeterInterface
from deepclustering.model import Model, ZeroGradientBackwardStep
from deepclustering.trainer import _Trainer
from deepclustering.utils import (
    tqdm_,
    simplex,
    tqdm,
    dict_filter,
    nice_dict,
    assert_list,
)
from deepclustering.utils.classification.assignment_mapping import (
    flat_acc,
    hungarian_match,
)
from torch import nn, Tensor
from torch.utils.data import DataLoader

from RegHelper import pred_histgram, VATModuleInterface, MixUp


class VATReg:

    def __init__(self, VAT_params: Dict[str, Union[str, float]] = {"eps": 10}, MeterInterface=None) -> None:
        super().__init__()

        self.VAT_params = VAT_params
        self.vat_module = VATModuleInterface(VAT_params)
        self.MeterInterface = MeterInterface
        if self.MeterInterface:
            self.MeterInterface.register_new_meter("train_adv", AverageValueMeter())

    def _vat_regularization(self, model: Model, img: Tensor, head="B") -> Tuple[Tensor, Tensor, Tensor]:
        vat_loss, adv_image, noise = self.vat_module(model, img, head=head)
        if self.MeterInterface:
            self.MeterInterface["train_adv"].add(vat_loss.item())
        return vat_loss, adv_image, noise


class GeoReg:

    def __init__(self) -> None:
        super().__init__()
        self.kl_div = KL_div(reduce=True)

    def _geo_regularization(self, tf1_pred_simplex, tf2_pred_simplex) -> Tensor:
        """
        :param tf1_pred_simplex: basic
        :param tf2_pred_simplex: advanced
        :return:
        """
        assert (
                assert_list(simplex, tf1_pred_simplex)
                and assert_list(simplex, tf2_pred_simplex)
                and tf1_pred_simplex.__len__() == tf2_pred_simplex.__len__()
        ), f"Error on tf1 and tf2 predictions."
        batch_loss: List[torch.Tensor] = []  # type: ignore
        for subhead in range(tf1_pred_simplex.__len__()):
            _loss = self.kl_div(
                tf2_pred_simplex[subhead], tf1_pred_simplex[subhead].detach()
            )
            batch_loss.append(_loss)
        batch_loss: torch.Tensor = sum(batch_loss) / len(batch_loss)  # type:ignore
        return batch_loss


class MixupReg:

    def __init__(self) -> None:
        super().__init__()
        self.mixup_module = MixUp(self.device, num_classes=self.model.arch_dict["output_k_B"])
        self.kl_div = KL_div(reduce=True)

    def _mixup_image_pred_index(self, tf1_image, tf1_pred, tf2_image, tf2_pred) -> Tuple[Tensor, Tensor, Tensor]:
        """
        There the input predictions are simplexes instead of list of simplexes
        """
        assert simplex(tf1_pred) and simplex(tf2_pred)
        mixup_img, mixup_label, mixup_index = self.mixup_module(
            tf1_image, tf1_pred, tf2_image, tf2_pred
        )
        return mixup_img, mixup_label, mixup_index


class ClusteringGeneralTrainer(_Trainer):
    # project save dirs for training statistics
    RUN_PATH = str(Path(__file__).parent.parent / "runs")
    ARCHIVE_PATH = str(Path(__file__).parent.parent / "archives")

    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module = None,
            max_epoch: int = 100,
            save_dir: str = "ClusteringGeneralTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,  # both IIC and IMSAT may need this sobel filter
            config: dict = None,
            **kwargs,
    ) -> None:
        super().__init__(
            model,
            None,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            config,
            **kwargs,
        )  # type: ignore
        assert (self.train_loader is None), self.train_loader  # discard the original self.train_loader
        self.train_loader_A = train_loader_A  # trainer for head_A
        self.train_loader_B = train_loader_B  # trainer for head B
        self.head_control_params: OrderedDict = OrderedDict(head_control_params)
        assert criterion, criterion
        self.criterion = criterion
        self.criterion.to(self.device)
        self.use_sobel = use_sobel
        if self.use_sobel:
            self.sobel = SobelProcess(include_origin=False)
            self.sobel.to(self.device)  # sobel filter return a tensor (bn, 1, w, h)

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        """
        basic meters to record clustering results, specifically for multi-subheads.
        :return:
        """
        METER_CONFIG = {
            "val_average_acc": AverageValueMeter(),
            "val_best_acc": AverageValueMeter(),
            "val_worst_acc": AverageValueMeter(),
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return [["val_average_acc_mean", "val_best_acc_mean", "val_worst_acc_mean"]]

    @property
    def _training_report_dict(self) -> Dict[str, float]:
        return {}  # to override

    @property
    def _eval_report_dict(self) -> Dict[str, float]:
        """
        return validation report dict
        :return:
        """
        report_dict = {
            "average_acc": self.METERINTERFACE.val_average_acc.summary()["mean"],
            "best_acc": self.METERINTERFACE.val_best_acc.summary()["mean"],
            "worst_acc": self.METERINTERFACE.val_worst_acc.summary()["mean"],
        }
        report_dict = dict_filter(report_dict)
        return report_dict

    def start_training(self):
        """
        main function to call for training
        :return:
        """
        for epoch in range(self._start_epoch, self.max_epoch):
            self._train_loop(
                train_loader_A=self.train_loader_A,
                train_loader_B=self.train_loader_B,
                epoch=epoch,
                head_control_param=self.head_control_params,
            )
            with torch.no_grad():
                current_score = self._eval_loop(self.val_loader, epoch)

            # update meters
            self.METERINTERFACE.step()
            # update model scheduler
            self.model.schedulerStep()
            # save meters and checkpoints
            SUMMARY = self.METERINTERFACE.summary()
            SUMMARY.to_csv(self.save_dir / f"wholeMeter.csv")
            # draw traing curves
            self.drawer.draw(SUMMARY)
            # save last.pth and/or best.pth based on current_score
            self.save_checkpoint(self.state_dict, epoch, current_score)
        # close tf.summary_writer
        time.sleep(3)
        self.writer.close()

    def _train_loop(
            self,
            train_loader_A: DataLoader = None,
            train_loader_B: DataLoader = None,
            epoch: int = None,
            mode: ModelMode = ModelMode.TRAIN,
            head_control_param: OrderedDict = None,
            *args,
            **kwargs,
    ) -> None:
        """
        :param train_loader_A:
        :param train_loader_B:
        :param epoch:
        :param mode:
        :param head_control_param:
        :param args:
        :param kwargs:
        :return: None
        """
        # robustness asserts
        assert isinstance(train_loader_B, DataLoader) and isinstance(
            train_loader_A, DataLoader
        )
        assert (head_control_param and head_control_param.__len__() > 0), \
            f"`head_control_param` must be provided, given {head_control_param}."
        assert set(head_control_param.keys()) <= {"A", "B", }, \
            f"`head_control_param` key must be in `A` or `B`, given {set(head_control_param.keys())}"
        for k, v in head_control_param.items():
            assert k in ("A", "B"), (
                f"`head_control_param` key must be in `A` or `B`,"
                f" given{set(head_control_param.keys())}"
            )
            assert isinstance(v, int) and v >= 0, f"Iteration for {k} must be >= 0."
        # set training mode
        self.model.set_mode(mode)
        assert (
            self.model.training
        ), f"Model should be in train() model, given {self.model.training}."
        assert len(train_loader_B) == len(train_loader_A), (
            f'The length of the train_loaders should be the same,"'
            f"given `len(train_loader_A)`:{len(train_loader_A)} and `len(train_loader_B)`:{len(train_loader_B)}."
        )

        for head_name, head_iterations in head_control_param.items():
            assert head_name in ("A", "B"), head_name
            train_loader = eval(f"train_loader_{head_name}")  # change the dataset for different head
            for head_epoch in range(head_iterations):
                # given one head, one iteration in this head, and one train_loader.
                train_loader_: tqdm = tqdm_(train_loader)  # reinitialize the train_loader
                train_loader_.set_description(
                    f"Training epoch: {epoch} head:{head_name}, head_epoch:{head_epoch + 1}/{head_iterations}"
                )
                for batch, image_labels in enumerate(train_loader_):
                    images, *_ = list(zip(*image_labels))
                    # extract tf1_images, tf2_images and put then to self.device
                    tf1_images = torch.cat(tuple([images[0] for _ in range(len(images) - 1)]), dim=0).to(self.device)
                    tf2_images = torch.cat(tuple(images[1:]), dim=0).to(self.device)
                    assert tf1_images.shape == tf2_images.shape, f"`tf1_images` should have the same size as `tf2_images`," \
                        f"given {tf1_images.shape} and {tf2_images.shape}."
                    # if images are processed with sobel filters
                    if self.use_sobel:
                        tf1_images = self.sobel(tf1_images)
                        tf2_images = self.sobel(tf2_images)
                        assert tf1_images.shape == tf2_images.shape
                    # Here you have two kinds of geometric transformations
                    # todo: functions to be overwritten
                    batch_loss = self._trainer_specific_loss(tf1_images, tf2_images, head_name)
                    # update model with self-defined context manager support Apex module
                    with ZeroGradientBackwardStep(batch_loss, self.model) as loss:
                        loss.backward()
                    # write value to tqdm module for system monitoring
                    report_dict = self._training_report_dict
                    train_loader_.set_postfix(report_dict)
        # for tensorboard recording
        self.writer.add_scalar_with_tag("train", report_dict, epoch)
        # for std recording
        print(f"Training epoch: {epoch} : {nice_dict(report_dict)}")

    def _eval_loop(
            self,
            val_loader: DataLoader = None,
            epoch: int = 0,
            mode: ModelMode = ModelMode.EVAL,
            return_soft_predict=False,
            *args,
            **kwargs,
    ) -> float:
        assert isinstance(val_loader, DataLoader)  # make sure a validation loader is passed.
        self.model.set_mode(mode)  # set model to be eval mode, by default.
        # make sure the model is in eval mode.
        assert (not self.model.training), f"Model should be in eval model in _eval_loop, given {self.model.training}."
        val_loader_: tqdm = tqdm_(val_loader)
        # prediction initialization with shape: (num_sub_heads, num_samples)
        preds = torch.zeros(self.model.arch_dict["num_sub_heads"],
                            val_loader.dataset.__len__(),
                            dtype=torch.long,
                            device=self.device)
        # soft_prediction initialization with shape (num_sub_heads, num_sample, num_classes)
        if return_soft_predict:
            soft_preds = torch.zeros(self.model.arch_dict["num_sub_heads"],
                                     val_loader.dataset.__len__(),
                                     self.model.arch_dict["output_k_B"],
                                     dtype=torch.float,
                                     device=torch.device("cpu"))  # I put it into cpu
        # target initialization with shape: (num_samples)
        target = torch.zeros(val_loader.dataset.__len__(), dtype=torch.long, device=self.device)
        # begin index
        slice_done = 0
        subhead_accs = []
        val_loader_.set_description(f"Validating epoch: {epoch}")
        for batch, image_labels in enumerate(val_loader_):
            images, gt, *_ = list(zip(*image_labels))
            # only take the tf3 image and gts, put them to self.device
            images, gt = images[0].to(self.device), gt[0].to(self.device)
            # if use sobel filter
            if self.use_sobel:
                images = self.sobel(images)
            # using default head_B for inference, _pred should be a list of simplex by default.
            _pred = self.model.torchnet(images, head="B")
            assert assert_list(simplex, _pred), "pred should be a list of simplexes."
            assert _pred.__len__() == self.model.arch_dict["num_sub_heads"]
            # slice window definition
            bSlicer = slice(slice_done, slice_done + images.shape[0])
            for subhead in range(self.model.arch_dict["num_sub_heads"]):
                # save predictions for each subhead for each batch
                preds[subhead][bSlicer] = _pred[subhead].max(1)[1]
                if return_soft_predict:
                    soft_preds[subhead][bSlicer] = _pred[subhead]
            # save target for each batch
            target[bSlicer] = gt
            # update slice index
            slice_done += gt.shape[0]
        # make sure that all the dataset has been done. Errors will raise if dataloader.drop_last=True
        assert slice_done == val_loader.dataset.__len__(), "Slice not completed."
        for subhead in range(self.model.arch_dict["num_sub_heads"]):
            # remap pred for each head and compare with target to get subhead_acc
            reorder_pred, remap = hungarian_match(
                flat_preds=preds[subhead],
                flat_targets=target,
                preds_k=self.model.arch_dict["output_k_B"],
                targets_k=self.model.arch_dict["output_k_B"],
            )
            _acc = flat_acc(reorder_pred, target)
            subhead_accs.append(_acc)
            # record average acc
            self.METERINTERFACE.val_average_acc.add(_acc)

            if return_soft_predict:
                soft_preds[subhead][:, list(remap.values())] = soft_preds[subhead][:, list(remap.keys())]
                assert torch.allclose(soft_preds[subhead].max(1)[1], reorder_pred.cpu())

        # record best acc
        self.METERINTERFACE.val_best_acc.add(max(subhead_accs))
        # record worst acc
        self.METERINTERFACE.val_worst_acc.add(min(subhead_accs))
        report_dict = self._eval_report_dict
        # record results for std
        print(f"Validating epoch: {epoch} : {nice_dict(report_dict)}")
        # record results for tensorboard
        self.writer.add_scalar_with_tag("val", report_dict, epoch)
        # using multithreads to call histogram interface of tensorboard.
        pred_histgram(self.writer, preds, epoch=epoch)
        # return the current score to save the best checkpoint.
        if return_soft_predict:
            return self.METERINTERFACE.val_best_acc.summary()["mean"], (
                target.cpu(), soft_preds[np.argmax(subhead_accs)])  # type ignore

        return self.METERINTERFACE.val_best_acc.summary()["mean"]

    def _trainer_specific_loss(self, tf1_images: Tensor, tf2_images: Tensor, head_name: str):
        """
        functions to be overrided
        :param tf1_images: basic transformed images with device = self.device
        :param tf2_images: advanced transformed image with device = self.device
        :param head_name: head name for model inference
        :return: loss tensor to call .backward()
        """

        raise NotImplementedError

    # using IIC method to show the evaluation, only for MNIST dataset
    def save_plot(self):
        assert self.val_loader.dataset_name == "mnist", \
            f"save tsne plot is only implemented for MNIST dataset, given {self.val_loader.dataset_name}."

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

    def draw_tsne(self, val_loader, epoch=0):
        assert val_loader.dataset_name == "mnist", \
            f"save tsne plot is only implemented for MNIST dataset, given {self.val_loader.dataset_name}."
        from deepclustering.arch.classification.IIC.net6c_two_head import ClusterNet6cTwoHead
        assert isinstance(self.model.torchnet,
                          ClusterNet6cTwoHead), f"self.model must be ClusterNet6cTwoHead, given {self.model}"

        def _draw_features_and_targets(val_loader):
            print(f"Feature generating.")
            features = []
            targets = []

            def hook(module, input, output):
                features.append(output.cpu().detach())

            handler = self.model.torchnet.trunk.register_forward_hook(hook)
            for batch, image_labels in enumerate(val_loader):
                images, gt, *_ = list(zip(*image_labels))
                # only take the tf3 image and gts, put them to self.device
                images, gt = images[0].to(self.device), gt[0].to(self.device)
                # if use sobel filter
                if self.use_sobel:
                    images = self.sobel(images)
                # using default head_B for inference, _pred should be a list of simplex by default.
                _pred = self.model.torchnet(images, head="B")
                targets.append(gt.cpu())
                if len(targets) > 20:
                    break
            features = torch.cat(features, 0)
            targets = torch.cat(targets, 0)
            handler.remove()
            print("Feature Generation end.")
            return features, targets

        features, targets = _draw_features_and_targets(val_loader)
        from deepclustering.decorator import TimeBlock
        with TimeBlock() as time:
            self.writer.add_embedding(mat=features, metadata=targets, global_step=epoch)
        print(time.cost)
