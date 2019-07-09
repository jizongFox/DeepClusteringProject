from typing import List, Union, Dict

import torch
from deepclustering.loss.IID_losses import IIDLoss
from deepclustering.meters import AverageValueMeter
from deepclustering.model import Model
from deepclustering.utils import (
    simplex,
    dict_filter,
    assert_list,
)
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from RegHelper import VATModuleInterface, MixUp
from .clustering_trainer import ClusteringGeneralTrainer


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
        """
        IIC trainer support multihead training
        :param head_control_params
            self.head_control_params={"A":1,"B"=2}
        """
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
        columns_to_draw = ["train_head_A_mean", "train_head_B_mean"] + columns_to_draw  # type:ignore
        return columns_to_draw

    @property
    def _training_report_dict(self):
        report_dict = {
            "train_head_A": self.METERINTERFACE["train_head_A"].summary()["mean"],
            "train_head_B": self.METERINTERFACE["train_head_B"].summary()["mean"],
        }
        return dict_filter(report_dict)

    def _trainer_specific_loss(self, tf1_images: Tensor, tf2_images: Tensor, head_name: str):
        """
        IIC loss for two types of transformations
        :param tf1_images: basic transformed images, with device = self.device
        :param tf2_images: advance transformed images, with device = self.device
        :param head_name: head_name
        :return: loss tensor to call .backward()
        """
        tf1_pred_simplex = self.model.torchnet(tf1_images, head=head_name)
        tf2_pred_simplex = self.model.torchnet(tf2_images, head=head_name)
        assert (
                assert_list(simplex, tf1_pred_simplex)
                and assert_list(simplex, tf2_pred_simplex)
                and tf1_pred_simplex.__len__() == tf2_pred_simplex.__len__()
        ), f"Error on tf1 and tf2 predictions."

        batch_loss: List[torch.Tensor] = []  # type: ignore
        for subhead in range(tf1_pred_simplex.__len__()):
            _loss, _loss_no_lambda = self.criterion(
                tf1_pred_simplex[subhead], tf2_pred_simplex[subhead]
            )
            batch_loss.append(_loss)
        batch_loss: torch.Tensor = sum(batch_loss) / len(batch_loss)  # type:ignore
        self.METERINTERFACE[f"train_head_{head_name}"].add(
            -batch_loss.item()
        )  # type: ignore

        return batch_loss


class IICVATTrainer(IICGeoTrainer):
    """
    We add the VAT module to create a set of VAT examples. These VAT examples replace tf2_images
    so that the MI takes tf1_images and VAT(tf1_images).
    No tf2_images are used in this trainer.
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
        self.VAT_module = VATModuleInterface(VAT_params)

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        # just replace the tf2_image with VAT generated images
        _, tf2_images, _ = self.VAT_module(self.model.torchnet, tf1_images)
        assert (not tf2_images.requires_grad) and (not tf1_images.requires_grad)
        batch_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        return batch_loss


class IICGeoVATTrainer(IICVATTrainer):
    """
    Use the MI between (tf1_images, VAT(tf1_images)+tf2_image))
    tf2_images are the original tf2_images with advanced transformation.
    for mathematic simplication, MI(a,b+c)>=MI(a,b)+MI(a,c), we maximize the lower bound
    of the MI by maximizing MI(tf1_images, tf2_images) + MI (tf1_images, VAT(tf1_images))
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
        # VAT_loss
        vat_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        # here the tf2_images is untouched
        # original Geo loss
        geo_loss = IICGeoTrainer._trainer_specific_loss(
            self, tf1_images, tf2_images, head_name
        )
        return vat_loss + geo_loss


class IICMixupTrainer(IICGeoTrainer):
    """
    This trainer is to replace tf2_images by a mixup image
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
