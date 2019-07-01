"""
This is the trainer general clustering trainer
"""
from collections import OrderedDict
from pathlib import Path
from typing import List, Union, Dict, Any

import matplotlib
import torch
from deepclustering import ModelMode
from deepclustering.augment.pil_augment import SobelProcess
from deepclustering.loss.IID_losses import IIDLoss
from deepclustering.loss.IMSAT_loss import MultualInformaton_IMSAT
from deepclustering.loss.loss import KL_div
from deepclustering.meters import AverageValueMeter, MeterInterface
from deepclustering.model import Model, ZeroGradientBackwardStep
from deepclustering.trainer import _Trainer
from deepclustering.utils import tqdm_, simplex, tqdm, dict_filter, nice_dict, assert_list
from deepclustering.utils.classification.assignment_mapping import (
    flat_acc,
    hungarian_match,
)
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from RegHelper import VATModuleInterface, MixUp, pred_histgram

matplotlib.use("agg")


class ClusteringGeneralTrainer(_Trainer):
    RUN_PATH = str(Path(__file__).parent / "runs")
    ARCHIVE_PATH = str(Path(__file__).parent / "archives")

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
            use_sobel: bool = False,
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
        assert (
                self.train_loader is None
        ), self.train_loader  # discard the original self.train_loader
        self.train_loader_A = train_loader_A
        self.train_loader_B = train_loader_B
        self.head_control_params: OrderedDict = OrderedDict(head_control_params)
        assert criterion, criterion
        self.criterion = criterion
        self.criterion.to(self.device)
        self.use_sobel = use_sobel
        if self.use_sobel:
            self.sobel = SobelProcess(include_origin=False)
            self.sobel.to(self.device)

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        METER_CONFIG = {
            "val_average_acc": AverageValueMeter(),
            "val_best_acc": AverageValueMeter(),
            "val_worst_acc": AverageValueMeter(),
        }
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return [["val_average_acc_mean", "val_best_acc_mean", "val_worst_acc_mean"]]

    @property
    def _training_report_dict(self):
        return {}

    @property
    def _eval_report_dict(self):
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
            self.METERINTERFACE.step()
            self.model.schedulerStep()
            # save meters and checkpoints
            SUMMARY = self.METERINTERFACE.summary()
            SUMMARY.to_csv(self.save_dir / f"wholeMeter.csv")
            self.drawer.draw(SUMMARY)
            self.save_checkpoint(self.state_dict, epoch, current_score)
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
        assert isinstance(train_loader_B, DataLoader) and isinstance(
            train_loader_A, DataLoader
        )
        assert (
                head_control_param and head_control_param.__len__() > 0
        ), f"`head_control_param` must be provided, given {head_control_param}."
        assert set(head_control_param.keys()) <= {
            "A",
            "B",
        }, f"`head_control_param` key must be in `A` or `B`, given {set(head_control_param.keys())}"
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
            train_loader = eval(
                f"train_loader_{head_name}"
            )  # change the datasets for different head
            for head_epoch in range(head_iterations):
                # given one head, one iteration in this head, and one train_loader.
                train_loader_: tqdm = tqdm_(
                    train_loader
                )  # reinitialize the train_loader
                train_loader_.set_description(
                    f"Training epoch: {epoch} head:{head_name}, head_epoch:{head_epoch + 1}/{head_iterations}"
                )
                # time_before = time.time()
                for batch, image_labels in enumerate(train_loader_):
                    images, *_ = list(zip(*image_labels))
                    tf1_images = torch.cat(
                        tuple([images[0] for _ in range(images.__len__() - 1)]), dim=0
                    ).to(self.device)
                    tf2_images = torch.cat(tuple(images[1:]), dim=0).to(self.device)
                    if self.use_sobel:
                        tf1_images = self.sobel(tf1_images)
                        tf2_images = self.sobel(tf2_images)
                    assert tf1_images.shape == tf2_images.shape
                    # Here you have two kinds of geometric transformations
                    batch_loss = self._trainer_specific_loss(
                        tf1_images, tf2_images, head_name
                    )

                    with ZeroGradientBackwardStep(batch_loss, self.model) as loss:
                        loss.backward()
                    report_dict = self._training_report_dict
                    train_loader_.set_postfix(report_dict)
        self.writer.add_scalar_with_tag("train", report_dict, epoch)
        print(f"Training epoch: {epoch} : {nice_dict(report_dict)}")

    def _eval_loop(
            self,
            val_loader: DataLoader = None,
            epoch: int = 0,
            mode: ModelMode = ModelMode.EVAL,
            *args,
            **kwargs,
    ) -> float:
        assert isinstance(val_loader, DataLoader)
        self.model.set_mode(mode)
        assert (
            not self.model.training
        ), f"Model should be in eval model in _eval_loop, given {self.model.training}."
        val_loader_: tqdm = tqdm_(val_loader)
        preds = torch.zeros(
            self.model.arch_dict["num_sub_heads"],
            val_loader.dataset.__len__(),
            dtype=torch.long,
            device=self.device,
        )
        target = torch.zeros(
            val_loader.dataset.__len__(), dtype=torch.long, device=self.device
        )
        slice_done = 0
        subhead_accs = []
        val_loader_.set_description(f"Validating epoch: {epoch}")
        for batch, image_labels in enumerate(val_loader_):
            images, gt, *_ = list(zip(*image_labels))
            images, gt = images[0].to(self.device), gt[0].to(self.device)
            if self.use_sobel:
                images = self.sobel(images)
            _pred = self.model.torchnet(images, head="B")
            assert _pred.__len__() == self.model.arch_dict["num_sub_heads"]
            assert simplex(_pred[0]), f"pred should be normalized, given {_pred[0]}."
            bSlicer = slice(slice_done, slice_done + images.shape[0])
            for subhead in range(self.model.arch_dict["num_sub_heads"]):
                preds[subhead][bSlicer] = _pred[subhead].max(1)[1]
                target[bSlicer] = gt

            slice_done += gt.shape[0]
        assert slice_done == val_loader.dataset.__len__(), "Slice not completed."
        for subhead in range(self.model.arch_dict["num_sub_heads"]):
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
        # record best acc
        self.METERINTERFACE.val_best_acc.add(max(subhead_accs))
        # record worst acc
        self.METERINTERFACE.val_worst_acc.add(min(subhead_accs))
        report_dict = self._eval_report_dict
        print(f"Validating epoch: {epoch} : {nice_dict(report_dict)}")
        self.writer.add_scalar_with_tag("val", report_dict, epoch)
        pred_histgram(self.writer, preds, epoch=epoch)

        return self.METERINTERFACE.val_best_acc.summary()["mean"]

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        raise NotImplementedError


class IMSATAbstractTrainer(ClusteringGeneralTrainer):
    """
    This trainer is to implement MI(X,P)+Reg in _train_specific_loss method
    self.criterion = MI
    self.distance = KL
    without implement Reg
    In IMSAT, the loss usually only takes the basic transformed image tf1 without touching tf2
    """

    def __init__(self, model: Model, train_loader_A: DataLoader, train_loader_B: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "IMSATAbstractTrainer", checkpoint_path: str = None,
                 device="cpu", head_control_params: Dict[str, int] = {"B": 1}, use_sobel: bool = False,
                 config: dict = None, MI_params: dict = {}, **kwargs) -> None:
        super().__init__(
            model,
            train_loader_A,
            train_loader_B,
            val_loader,
            MultualInformaton_IMSAT(**MI_params),
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            head_control_params,
            use_sobel,
            config,
            **kwargs,
        )

        self.kl_div = KL_div(reduce=True)

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        colum_to_draw = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_mi", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("train_entropy", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("train_centropy", AverageValueMeter())
        colum_to_draw = ["train_mi_mean",
                         "train_entropy_mean",
                         "train_centropy_mean"] + colum_to_draw  # type: ignore
        return colum_to_draw

    @property
    def _training_report_dict(self):
        report_dict = {
            "mi": self.METERINTERFACE["train_mi"].summary()["mean"],
            "entropy": self.METERINTERFACE["train_entropy"].summary()["mean"],
            "centropy": self.METERINTERFACE["train_centropy"].summary()["mean"],
        }
        return dict_filter(report_dict)

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ) -> Tensor:
        assert (
                head_name == "B"
        ), "Only head B is supported in IMSAT, try to set head_control_parameter as {`B`:1}"
        # only tf1_images are needed
        tf1_images = tf1_images.to(self.device)
        tf1_pred_simplex = self.model.torchnet(tf1_images, head=head_name)
        # tf2_pred_simplex = self.model.torchnet(tf2_images, head=head_name)
        # assert simplex(tf1_pred_simplex[0]) and tf1_pred_simplex.__len__() == tf2_pred_simplex.__len__()
        batch_loss: List[torch.Tensor] = []  # type: ignore
        entropies: List[torch.Tensor] = []
        centropies: List[torch.Tensor] = []
        for pred in tf1_pred_simplex:
            mi, (entropy, centropy) = self.criterion(pred)
            batch_loss.append(mi)
            entropies.append(entropy)
            centropies.append(centropy)
        batch_loss: Tensor = sum(batch_loss) / len(batch_loss)  # type: ignore
        entropies: Tensor = sum(entropies) / len(entropies)  # type: ignore
        centropies: Tensor = sum(centropies) / len(centropies)  # type: ignore
        self.METERINTERFACE["train_mi"].add(batch_loss.item())
        self.METERINTERFACE["train_entropy"].add(entropies.item())
        self.METERINTERFACE["train_centropy"].add(centropies.item())
        reg_loss = self._regulaze(tf1_images, tf2_images, tf1_pred_simplex, head_name)
        return -batch_loss * 0.1 + reg_loss

    def _regulaze(
            self,
            images: Tensor,
            tf_images: Tensor,
            img_pred_simplex: List[Tensor],
            head_name: str = "B",
    ) -> Tensor:
        return torch.Tensor([0]).to(self.device)


class IMSATVATTrainer(IMSATAbstractTrainer):
    """
    implement VAT in the regularization method
    You will never use mi in IMSAT framework.
    """

    def __init__(self, model: Model, train_loader_A: DataLoader, train_loader_B: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "IMSATAbstractTrainer", checkpoint_path: str = None,
                 device="cpu", head_control_params: Dict[str, int] = {"B": 1}, use_sobel: bool = False,
                 config: Dict[str, Union[int, float, str, Dict[str, Any]]] = None,
                 MI_params: Dict[str, Union[int, float, str]] = {},
                 VAT_params: Dict[str, Union[int, float, str]] = {},
                 **kwargs: Dict[str, Union[int, float, str]]) -> None:
        if VAT_params.get("name"):
            assert VAT_params.get("name") == "kl", (
                f"In IMSAT framework, KL distance is the only to be supported, "
                f"given {VAT_params.get('name')}."
            )
        super().__init__(model, train_loader_A, train_loader_B, val_loader, max_epoch, save_dir, checkpoint_path,
                         device, head_control_params, use_sobel, config, MI_params, **kwargs)
        self.reg_module = VATModuleInterface(VAT_params)

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        colums = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_adv", AverageValueMeter())
        colums.insert(-1, "train_adv_mean")
        return colums

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update(
            {"adv": self.METERINTERFACE["train_adv"].summary()["mean"]}
        )
        return dict_filter(report_dict)

    def _regulaze(
            self,
            images: Tensor,
            tf_images: Tensor,
            img_pred_simplex: List[Tensor],
            head_name="B",
    ) -> Tensor:
        reg_loss, *_ = self.reg_module(self.model.torchnet, images, head=head_name)
        self.METERINTERFACE["train_adv"].add(reg_loss.item())
        return reg_loss


class IMSATVATGeoTrainer(IMSATVATTrainer):

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        columns = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_geo", AverageValueMeter())
        return ["train_geo_mean"] + columns

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update(
            {"geo": self.METERINTERFACE["train_geo"].summary()["mean"]}
        )
        return dict_filter(report_dict)

    def _regulaze(
            self,
            images: Tensor,
            tf_images: Tensor,
            img_pred_simplex: List[Tensor],
            head_name="B",
    ) -> Tensor:
        vat_loss = super()._regulaze(images, tf_images, img_pred_simplex, head_name)

        tf_images = tf_images.to(self.device)
        tf_pred_simplex = self.model.torchnet(tf_images, head=head_name)
        assert assert_list(simplex, tf_pred_simplex) and len(tf_pred_simplex) == len(img_pred_simplex)
        # kl div:
        geo_losses: List[Tensor] = []
        for subhead, (tf1_pred, tf2_pred) in enumerate(zip(img_pred_simplex, tf_pred_simplex)):
            assert simplex(tf1_pred) and simplex(tf2_pred)
            geo_losses.append(self.kl_div(tf2_pred, tf1_pred.detach()))
        geo_losses: Tensor = sum(geo_losses) / len(geo_losses)
        self.METERINTERFACE["train_geo"].add(geo_losses.item())
        return vat_loss + geo_losses


class IMSATMixupTrainer(IMSATVATTrainer):
    """
    implement Mixup in the regularization method
    You will use KL as the distance function to link the two
    """

    def __init__(self, model: Model, train_loader_A: DataLoader, train_loader_B: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "IMSATAbstractTrainer", checkpoint_path: str = None,
                 device="cpu", head_control_params: Dict[str, int] = {"B": 1}, use_sobel: bool = False,
                 config: Dict[str, Union[int, float, str, Dict[str, Any]]] = None,
                 MI_params: Dict[str, Union[int, float, str]] = {}, **kwargs) -> None:
        super().__init__(model, train_loader_A, train_loader_B, val_loader, max_epoch, save_dir, checkpoint_path,
                         device, head_control_params, use_sobel, config, MI_params, **kwargs)
        # override the regularzation module
        self.reg_module = MixUp(
            device=self.device, num_classes=self.model.arch_dict["output_k_B"]
        )

    def _regulaze(
            self,
            images: Tensor,
            tf_images: Tensor,
            img_pred_simplex: List[Tensor],
            head_name="B",
    ) -> Tensor:
        # here just use the tf1_image to mixup
        # nothing with tf2_images
        assert len(images) == len(img_pred_simplex[0])
        reg_losses: List[Tensor] = []
        for subhead, tf1_pred in enumerate(img_pred_simplex):
            mixup_img, mixup_label, mixup_index = self.reg_module(
                images, tf1_pred, images.flip(0), tf1_pred.flip(0)
            )
            subhead_loss = self.kl_div(self.model(mixup_img)[subhead], mixup_label)
            reg_losses.append(subhead_loss)
        reg_losses: Tensor = sum(reg_losses) / len(reg_losses)
        self.METERINTERFACE["train_reg"].add(reg_losses.item())
        return reg_losses


class IMSATVATGeoMixupTrainer(IMSATVATGeoTrainer):
    def __init__(self, model: Model, train_loader_A: DataLoader, train_loader_B: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "IMSATAbstractTrainer", checkpoint_path: str = None,
                 device="cpu", head_control_params: Dict[str, int] = {"B": 1}, use_sobel: bool = False,
                 config: Dict[str, Union[int, float, str, Dict[str, Any]]] = None,
                 MI_params: Dict[str, Union[int, float, str]] = {},
                 **kwargs: Dict[str, Union[int, float, str]]) -> None:
        super().__init__(model, train_loader_A, train_loader_B, val_loader, max_epoch, save_dir, checkpoint_path,
                         device, head_control_params, use_sobel, config, MI_params, **kwargs)
        self.mixup_module = MixUp(
            device=self.device, num_classes=self.model.arch_dict["output_k_B"]
        )

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        cloumns = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_mixup", AverageValueMeter())
        cloumns.insert(2, "train_mixup_mean")
        return cloumns

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update(
            {"train_mixup": self.METERINTERFACE["train_mixup"].summary()["mean"]}
        )
        return dict_filter(report_dict)

    def _regulaze(
            self,
            images: Tensor,
            tf_images: Tensor,
            img_pred_simplex: List[Tensor],
            head_name="B",
    ) -> Tensor:
        vat_geo_loss = super()._regulaze(images, tf_images, img_pred_simplex, head_name)
        # here just use the tf1_image to mixup
        # nothing with tf2_images
        assert len(images) == len(img_pred_simplex[0])
        reg_losses: List[Tensor] = []
        for subhead, tf1_pred in enumerate(img_pred_simplex):
            mixup_img, mixup_label, mixup_index = self.mixup_module(
                images, tf1_pred, images.flip(0), tf1_pred.flip(0)
            )
            subhead_loss = self.kl_div(self.model(mixup_img)[subhead], mixup_label)
            reg_losses.append(subhead_loss)
        reg_losses: Tensor = sum(reg_losses) / len(reg_losses)
        self.METERINTERFACE["train_mixup"].add(reg_losses.item())
        return reg_losses + vat_geo_loss


class IICGeoTrainer(ClusteringGeneralTrainer):
    """
    This trainer is to add the IIC loss in the `_train_specific_loss` function.
    """

    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = "IICTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,
            config: dict = None,
            **kwargs,
    ) -> None:
        super().__init__(
            model,
            train_loader_A,
            train_loader_B,
            val_loader,
            IIDLoss(),
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            head_control_params,
            use_sobel,
            config,
            **kwargs,
        )

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        columns_to_draw = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_head_A", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("train_head_B", AverageValueMeter())
        columns_to_draw = [
                              "train_head_A_mean",
                              "train_head_B_mean",
                          ] + columns_to_draw  # type:ignore
        return columns_to_draw

    @property
    def _training_report_dict(self):
        report_dict = {
            "train_head_A": self.METERINTERFACE["train_head_A"].summary()["mean"],
            "train_head_B": self.METERINTERFACE["train_head_B"].summary()["mean"],
        }
        return dict_filter(report_dict)

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        if self.device not in (tf1_images.device, tf2_images.device):
            tf1_images, tf2_images = tf1_images.to(self.device), tf2_images.to(self.device)
        tf1_pred_simplex = self.model.torchnet(tf1_images, head=head_name)
        tf2_pred_simplex = self.model.torchnet(tf2_images, head=head_name)
        assert assert_list(simplex, tf1_pred_simplex) and assert_list(simplex, tf2_pred_simplex) and \
               tf1_pred_simplex.__len__() == tf2_pred_simplex.__len__(), f"Error on tf1 and tf2 predictions."

        batch_loss: List[torch.Tensor] = []  # type: ignore
        for subhead in range(tf1_pred_simplex.__len__()):
            _loss, _loss_no_lambda = self.criterion(
                tf1_pred_simplex[subhead], tf2_pred_simplex[subhead]
            )
            batch_loss.append(_loss)
        batch_loss: torch.Tensor = sum(batch_loss) / len(batch_loss)  # type:ignore
        self.METERINTERFACE[f"train_head_{head_name}"].add(-batch_loss.item())  # type: ignore

        return batch_loss


class IICVATTrainer(IICGeoTrainer):
    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = "IICTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,
            config: dict = None,
            VAT_params: Dict[str, Union[int, float, str]] = {"name": "kl"},
            **kwargs,
    ) -> None:
        super().__init__(
            model,
            train_loader_A,
            train_loader_B,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            head_control_params,
            use_sobel,
            config,
            **kwargs,
        )
        self.VAT_module = VATModuleInterface(
            {**VAT_params, **{"only_return_img": True}}
        )

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        # just replace the tf2_image with VAT generated images
        tf1_images = tf1_images.to(self.device)
        _, tf2_images, _ = self.VAT_module(self.model.torchnet, tf1_images)
        batch_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        return batch_loss


class IICGeoVATTrainer(IICVATTrainer):
    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = "IICTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,
            config: dict = None,
            VAT_params: Dict[str, Union[int, float, str]] = {"name": "kl"},
            **kwargs,
    ) -> None:
        super().__init__(
            model,
            train_loader_A,
            train_loader_B,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            head_control_params,
            use_sobel,
            config,
            VAT_params,
            **kwargs,
        )

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        vat_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        geo_loss = IICGeoTrainer._trainer_specific_loss(
            self, tf1_images, tf2_images, head_name
        )
        return vat_loss + geo_loss


class IICMixupTrainer(IICGeoTrainer):
    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = "IICTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,
            config: dict = None,
            **kwargs,
    ) -> None:
        super().__init__(
            model,
            train_loader_A,
            train_loader_B,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            head_control_params,
            use_sobel,
            config,
            **kwargs,
        )
        self.mixup_module = MixUp(
            device=self.device, num_classes=self.model.arch_dict["output_k_B"]
        )

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        # just replace tf2_images with mix_up generated images
        tf1_images = tf1_images.to(self.device)
        tf2_images, *_ = self.mixup_module(
            tf1_images,
            F.softmax(torch.randn(tf1_images.size(0), 2, device=self.device), 1),
            tf1_images.flip(0),
            F.softmax(torch.randn(tf1_images.size(0), 2, device=self.device), 1),
        )
        batch_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        return batch_loss


class IICGeoVATMixupTrainer(IICGeoTrainer):
    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = "IICVATMixupTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,
            config: dict = None,
            **kwargs,
    ) -> None:
        super().__init__(
            model,
            train_loader_A,
            train_loader_B,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            head_control_params,
            use_sobel,
            config,
            **kwargs,
        )
        self.mixup_module = MixUp(
            device=self.device, num_classes=self.model.arch_dict["output_k_B"]
        )

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        geo_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        mixup_loss = self.mixup_loss(tf1_images, tf2_images, head_name)
        vat_loss = self.vat_loss(tf1_images, tf2_images, head_name)
        return geo_loss + mixup_loss + vat_loss

    def mixup_loss(self, tf1_images: Tensor, tf2_images: Tensor, head_name: str):
        # just replace tf2_images with mix_up generated images
        tf1_images = tf1_images.to(self.device)
        tf2_images, *_ = self.mixup_module(
            tf1_images,
            F.softmax(torch.randn(tf1_images.size(0), 2, device=self.device), 1),
            tf1_images.flip(0),
            F.softmax(torch.randn(tf1_images.size(0), 2, device=self.device), 1),
        )
        batch_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        return batch_loss

    def vat_loss(self, tf1_images: Tensor, tf2_images: Tensor, head_name: str):
        # just replace the tf2_image with VAT generated images
        _, tf2_images, _ = self.VAT_module(self.model.torchnet, tf1_images)
        batch_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        return batch_loss
