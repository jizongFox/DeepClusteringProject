"""
This is the trainer general clustering trainer
"""
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Union, Dict, Tuple

import numpy as np
import torch
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
from termcolor import colored
from torch import nn, Tensor
from torch.utils.data import DataLoader

from RegHelper import pred_histgram, VATModuleInterface, MixUp


class GuassianAdder:
    """
    This is the transformation class to add gaussian noise on PyTorch Tensor images.
    """

    def __init__(self, gaussian_std) -> None:
        super().__init__()
        assert isinstance(gaussian_std, float), type(gaussian_std)
        self.gaussian_std = gaussian_std
        print(colored(f"Gaussian Noise Adder with std={gaussian_std}", "green"))

    def __call__(self, input_images: Tensor):
        b, c, h, w = input_images.shape  # here the input images should have 4 dimensions
        _noise = torch.randn_like(input_images, device=input_images.device,
                                  dtype=input_images.dtype) * self.gaussian_std
        return input_images + _noise


class TensorCutout:
    r"""
    This function remove a box by randomly choose one part within image Tensor
    """

    def __init__(
            self, min_box: int, max_box: int, pad_value: Union[int, float] = 0
    ) -> None:
        r"""
        :param min_box: minimal box size
        :param max_box: maxinmal box size
        """
        super().__init__()
        self.min_box = int(min_box)
        self.max_box = int(max_box)
        self.pad_value = pad_value

    def _cutout_per_image(self, image_tensor: Tensor) -> Tensor:
        """
        Here the input should be one tensor image.
        :param image_tensor:
        :return:
        """
        c, h, w = image_tensor.shape
        r_img_tensor = (
            image_tensor.copy()
            if isinstance(image_tensor, np.ndarray)
            else image_tensor.clone()
        )
        # find left, upper, right, lower
        box_sz = np.random.randint(self.min_box, self.max_box + 1)
        half_box_sz = int(np.floor(box_sz / 2.0))
        x_c = np.random.randint(half_box_sz, w - half_box_sz)
        y_c = np.random.randint(half_box_sz, h - half_box_sz)
        box = (
            x_c - half_box_sz,
            y_c - half_box_sz,
            x_c + half_box_sz,
            y_c + half_box_sz,
        )
        r_img_tensor[:, box[1]: box[3], box[0]: box[2]] = self.pad_value
        return r_img_tensor

    def __call__(self, img_tensors: Tensor) -> Tensor:
        assert isinstance(img_tensors, Tensor)
        b, c, h, w = img_tensors.shape
        r_img_tensors = torch.stack([self._cutout_per_image(img) for img in img_tensors], dim=0)
        return r_img_tensors


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
        _batch_loss: List[torch.Tensor] = []  # type: ignore
        for subhead in range(tf1_pred_simplex.__len__()):
            _loss = self.kl_div(
                tf2_pred_simplex[subhead], tf1_pred_simplex[subhead].detach()
            )
            _batch_loss.append(_loss)
        batch_loss: torch.Tensor = sum(_batch_loss) / len(_batch_loss)  # type:ignore
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


class GaussianReg:

    def __init__(self, gaussian_std: float = 0.1) -> None:
        super().__init__()
        self.gaussian_adder = GuassianAdder(gaussian_std)
        self.kl_div = KL_div(reduce=True)

    def _gaussian_regularization(self, model: Model, tf1_images, tf1_pred_simplex: List[Tensor], head_name="B") -> Tensor:

        """
        calculate predicton simplexes on gaussian noise tf1 images and the kl div of the original prediction simplex.
        :param tf1_images: tf1-transformed images
        :param tf1_pred_simplex: simplex list of tf1-transformed image prediction
        :return:  loss
        """
        _tf1_images_gaussian = self.gaussian_adder(tf1_images)
        _tf1_gaussian_simplex = model(_tf1_images_gaussian,head=head_name)
        assert assert_list(simplex, tf1_pred_simplex)
        assert assert_list(simplex, _tf1_gaussian_simplex)
        assert tf1_pred_simplex.__len__() == _tf1_gaussian_simplex.__len__()
        reg_loss = []
        for __tf1_simplex, __tf1_gaussian_simplex in zip(tf1_pred_simplex, _tf1_gaussian_simplex):
            reg_loss.append(self.kl_div(__tf1_gaussian_simplex, __tf1_simplex.detach()))
        return sum(reg_loss) / len(reg_loss)  # type: ignore


class CutoutReg:

    def __init__(self, min_box: int = 6, max_box: int = 12, pad_value: float = 0.5) -> None:
        super().__init__()
        self.tensorcutout = TensorCutout(
            min_box=min_box,
            max_box=max_box,
            pad_value=pad_value
        )
        print(
            colored(f"Initialize `Cutout` with max_box={max_box}, min_box={min_box}, pad_value={pad_value}.", "green"))
        self.kl_div = KL_div(reduce=True)

    def _cutout_regularization(self, model, tf1_images: Tensor, tf1_pred_simplex: List[Tensor], head_name="B") -> Tensor:
        _tf1_cutout_images = self._cutout_images(tf1_images)
        _tf1_cutout_pred_simplex = model(_tf1_cutout_images,head= head_name)
        _loss: List[Tensor] = []
        for head_num, (_tf1_cutout_pred, _tf1_pred) in enumerate(zip(_tf1_cutout_pred_simplex, tf1_pred_simplex)):
            _loss.append(self.kl_div(_tf1_cutout_pred, _tf1_pred.detach()))
        loss: Tensor = sum(_loss) / len(_loss)  # type: ignore
        return loss

    def _cutout_images(self, image):
        b, c, h, w = image.shape
        return self.tensorcutout(image)


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
            self.save_checkpoint(self.state_dict(), epoch, current_score)
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
