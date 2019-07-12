__all__ = ["IICMixup_RegTrainer", "IICGeoTrainer", "IICVATMixup_RegTrainer", "IICVAT_RegTrainer"]
from typing import Union, Dict, List

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

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ):
        # original_loss
        geo_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        # vat regularization
        vat_loss, *_ = self._vat_regularization(self.model.torchnet, tf1_images, head=head_name)
        return geo_loss + vat_loss


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
        MixupReg.__init__(self)

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
        return geo_loss + _reg_losses


# Vat+Mixup
class IICVATMixup_RegTrainer(IICMixup_RegTrainer, VATReg):
    def __init__(self, model: Model, train_loader_A: DataLoader, train_loader_B: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "IICTrainer", checkpoint_path: str = None, device="cpu",
                 head_control_params: Dict[str, int] = {"B": 1}, use_sobel: bool = False, config: dict = None,
                 VAT_params: Dict[str, Union[int, float, str]] = {"name": "kl", "eps": 10},
                 **kwargs) -> None:
        IICMixup_RegTrainer.__init__(self, model, train_loader_A, train_loader_B, val_loader, max_epoch, save_dir,
                                     checkpoint_path,
                                     device, head_control_params, use_sobel, config, **kwargs)
        VATReg.__init__(self, VAT_params)

    def _trainer_specific_loss(self, tf1_images: Tensor, tf2_images: Tensor, head_name: str):
        geo_mixup_loss = super()._trainer_specific_loss(tf1_images, tf2_images, head_name)
        vat_loss, *_ = self._vat_regularization(self.model.torchnet, tf1_images, head=head_name)
        return geo_mixup_loss + vat_loss
