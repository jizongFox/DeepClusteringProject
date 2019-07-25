__all__ = ["IICMixup_RegTrainer", "IICGeoTrainer", "IICVATMixup_RegTrainer", "IICVAT_RegTrainer",
           "IICVATVAT_RegTrainer", "IICVATMI_VATKL"]
from typing import Union, Dict, List
from copy import deepcopy as dcp
from deepclustering.meters import AverageValueMeter
from deepclustering.model import Model
from torch import Tensor
from torch.utils.data import DataLoader

from .clustering_trainer import VATReg, MixupReg
from .iic_trainer import IICGeoTrainer


# VAT
class IICVAT_RegTrainer(IICGeoTrainer, VATReg):
    """
    This is to regularize IIC Geo with other regularzation formula, such as VAT and Mixup. The only different is that
    here the thing is linked with KL_div.
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
            reg_weight: float = 0.05,
            **kwargs,
    ) -> None:
        IICGeoTrainer.__init__(self,
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
        VATReg.__init__(self, VAT_params)
        self.reg_weight = reg_weight
        print(f"reg_weight={reg_weight}")

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        columns = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_adv", AverageValueMeter())
        columns = ["train_adv_mean"] + columns
        return columns

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update({
            "adv": self.METERINTERFACE["train_adv"].summary()["mean"]
        })
        return report_dict

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        # original_loss
        geo_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        # vat regularization
        vat_loss, *_ = self._vat_regularization(self.model.torchnet, tf1_images, head=head_name)
        self.METERINTERFACE["train_adv"].add(vat_loss.item())
        return geo_loss + self.reg_weight * vat_loss


class IICVATVAT_RegTrainer(IICVAT_RegTrainer):
    """
    This is to implement MI(p(x), p(T(x)))+ self.weights*(KL(p(VAT(x)),p(x)) + KL(p(VAT(T(x))),p(T(x))))
    """

    def _trainer_specific_loss(self, tf1_images: Tensor, tf2_images: Tensor, head_name: str):
        iic_vat_reg_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        vat_loss_2, *_ = self._vat_regularization(self.model.torchnet, tf2_images, head=head_name)
        return iic_vat_reg_loss + self.reg_weight * vat_loss_2


# MIXUP
class IICMixup_RegTrainer(IICGeoTrainer, MixupReg):
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
            reg_weight: float = 0.05,
            **kwargs,
    ) -> None:
        IICGeoTrainer.__init__(self,
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
        self.reg_weight = reg_weight
        MixupReg.__init__(self)
        print(f"reg_weight={reg_weight}")

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        columns = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_mixup", AverageValueMeter())
        columns = ["train_mixup_mean"] + columns
        return columns

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update({
            "mixup": self.METERINTERFACE["train_mixup"].summary()["mean"]
        })
        return report_dict

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        # original loss
        geo_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        reg_losses: List[Tensor] = []
        img_pred_simplex = self.model.torchnet(tf1_images, head=head_name)
        for subhead, tf1_pred in enumerate(img_pred_simplex):
            mixup_img, mixup_label, mixup_index = self._mixup_image_pred_index(
                tf1_images, tf1_pred, tf1_images.flip(0), tf1_pred.flip(0)
            )
            subhead_loss = self.kl_div(self.model.torchnet(mixup_img, head=head_name)[subhead], mixup_label)
            reg_losses.append(subhead_loss)
        _reg_losses: Tensor = sum(reg_losses) / len(reg_losses)
        self.METERINTERFACE["train_mixup"].add(_reg_losses.item())
        return geo_loss + self.reg_weight * _reg_losses


# Vat+Mixup
class IICVATMixup_RegTrainer(IICMixup_RegTrainer, VATReg):
    def __init__(self, model: Model, train_loader_A: DataLoader, train_loader_B: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "IICTrainer", checkpoint_path: str = None, device="cpu",
                 head_control_params: Dict[str, int] = {"B": 1}, use_sobel: bool = False, config: dict = None,
                 VAT_params: Dict[str, Union[int, float, str]] = {"name": "kl", "eps": 10},
                 reg_weight=0.05,
                 **kwargs) -> None:
        IICMixup_RegTrainer.__init__(self, model, train_loader_A, train_loader_B, val_loader, max_epoch, save_dir,
                                     checkpoint_path,
                                     device, head_control_params, use_sobel, config, reg_weight, **kwargs)
        VATReg.__init__(self, VAT_params)

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        columns = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_adv", AverageValueMeter())
        columns = ["train_adv_mean"] + columns
        return columns

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update({
            "adv": self.METERINTERFACE["train_adv"].summary()["mean"]
        })
        return report_dict

    def _trainer_specific_loss(self, tf1_images: Tensor, tf2_images: Tensor, head_name: str):
        geo_mixup_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        vat_loss, *_ = self._vat_regularization(self.model.torchnet, tf1_images, head=head_name)
        self.METERINTERFACE["train_adv"].add(vat_loss.item())
        return geo_mixup_loss + vat_loss


# special case IICVAT_MI + VAT
class IICVATMI_VATKL(IICVAT_RegTrainer):

    def __init__(self, model: Model, train_loader_A: DataLoader, train_loader_B: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "IICTrainer", checkpoint_path: str = None, device="cpu",
                 head_control_params: Dict[str, int] = {"B": 1}, use_sobel: bool = False, config: dict = None,
                 VAT_params: Dict[str, Union[int, float, str]] = {"name": "kl"}, reg_weight: float = 0.05,
                 **kwargs) -> None:
        super().__init__(model, train_loader_A, train_loader_B, val_loader, max_epoch, save_dir, checkpoint_path,
                         device, head_control_params, use_sobel, config, VAT_params, reg_weight, **kwargs)
        from RegHelper import VATModuleInterface
        _VAT_params = dcp(VAT_params)
        _VAT_params["name"] = "mi"
        self.mi_vat_module = VATModuleInterface(_VAT_params)

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        # replace tf2_image from geo-transformed to adversarial images based on MI, then calling super() would be fine.
        _, tf2_images, _ = self.mi_vat_module(self.model.torchnet, tf1_images, head=head_name)
        assert tf1_images.shape == tf2_images.shape
        loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        return loss
