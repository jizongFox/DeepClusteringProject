__all__ = ["IMSATAbstractTrainer", "IMSATVATGeoMixupTrainer", "IMSATVATGeoTrainer", "IMSATVATTrainer",
           "IMSATMixupTrainer", "IMSATVATMixupTrainer"]
from typing import List, Union, Dict, Any

import torch
from deepclustering.loss.IMSAT_loss import MultualInformaton_IMSAT
from deepclustering.loss.loss import KL_div
from deepclustering.meters import AverageValueMeter
from deepclustering.model import Model
from deepclustering.utils import (
    simplex,
    dict_filter,
    assert_list,
)
from torch import Tensor
from torch.utils.data import DataLoader

from RegHelper import VATModuleInterface, MixUp
from .clustering_trainer import ClusteringGeneralTrainer


class IMSATAbstractTrainer(ClusteringGeneralTrainer):
    """
    This trainer is to implement MI(X,P)+Reg in _train_specific_loss method
    >>> self.criterion = MI
    >>> self.distance = KL
    without implement Reg
    In IMSAT, the loss usually only takes the basic transformed image tf1 without touching tf2
    """

    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = "IMSATAbstractTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,
            config: dict = None,
            MI_params: dict = {},
            **kwargs,
    ) -> None:
        """
        :param MI_params: M_params to pass to MultualInformaton_IMSAT module.
        >>> self.criterion = MultualInformaton_IMSAT(**MI_params)
        MI_params can be {mu:4} by default or {mu:16} if the entropy is weak.

        """
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
        """
        Initialize the meters by extending the father class with MI related meters.
        :return: [ "train_mi_mean", "train_entropy_mean", "train_centropy_mean", validation meters]
        """
        colum_to_draw = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_mi", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("train_entropy", AverageValueMeter())
        self.METERINTERFACE.register_new_meter("train_centropy", AverageValueMeter())
        colum_to_draw = [
                            "train_mi_mean",
                            "train_entropy_mean",
                            "train_centropy_mean",
                        ] + colum_to_draw  # type: ignore
        return colum_to_draw

    @property
    def _training_report_dict(self):
        """
        training related meters, including mi, entropy and Centropy.
        :return:
        """
        report_dict = {
            "mi": self.METERINTERFACE["train_mi"].summary()["mean"],
            "entropy": self.METERINTERFACE["train_entropy"].summary()["mean"],
            "centropy": self.METERINTERFACE["train_centropy"].summary()["mean"],
        }
        return dict_filter(report_dict)

    def _trainer_specific_loss(
            self, tf1_images: Tensor, tf2_images: Tensor, head_name: str
    ) -> Tensor:
        """
        MI+Reg implementation of MI loss on tf1_images. Reg is going to be overrided by children modules
        :param tf1_images: basic transformed images with device = self.device
        :param tf2_images: advanced transformed image with device = self.device
        :param head_name: head name for model inference
        :return: loss tensor to call .backward()
        """
        assert (head_name == "B"), "Only head B is supported in IMSAT, try to set head_control_parameter as {`B`:1}"
        # only tf1_images are needed
        tf1_pred_simplex = self.model.torchnet(tf1_images, head=head_name)
        assert assert_list(simplex, tf1_pred_simplex), "Prediction must be a list of simplexes."
        batch_loss: List[torch.Tensor] = []  # type: ignore
        entropies: List[torch.Tensor] = []
        centropies: List[torch.Tensor] = []
        for pred in tf1_pred_simplex:
            mi, (entropy, centropy) = self.criterion(pred)
            batch_loss.append(mi)
            entropies.append(entropy)
            centropies.append(centropy)
        # MI object function to be maximized.
        batch_loss: Tensor = sum(batch_loss) / len(batch_loss)  # type: ignore
        entropies: Tensor = sum(entropies) / len(entropies)  # type: ignore
        centropies: Tensor = sum(centropies) / len(centropies)  # type: ignore
        self.METERINTERFACE["train_mi"].add(batch_loss.item())
        self.METERINTERFACE["train_entropy"].add(entropies.item())
        self.METERINTERFACE["train_centropy"].add(centropies.item())
        # add regularizations such as VAT, Mixup, GEO or more.
        reg_loss = self._regulaze(tf1_images, tf2_images, tf1_pred_simplex, head_name)
        # decrease the importance of MI, based on the IMSAT chainer implementation.
        return -batch_loss * 0.1 + reg_loss

    def _regulaze(
            self,
            images: Tensor,
            tf_images: Tensor,
            img_pred_simplex: List[Tensor],
            head_name: str = "B",
    ) -> Tensor:
        """
        No Regularization is required.
        :return:
        """
        return torch.Tensor([0]).to(self.device)


class IMSATVATTrainer(IMSATAbstractTrainer):
    """
    implement VAT in the regularization method
    You will never use mi in IMSAT framework.
    """

    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = "IMSATAbstractTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,
            config: Dict[str, Union[int, float, str, Dict[str, Any]]] = None,
            MI_params: Dict[str, Union[int, float, str]] = {},
            VAT_params: Dict[str, Union[int, float, str]] = {},
            **kwargs: Dict[str, Union[int, float, str]],
    ) -> None:
        """
        IMSAT implementations
        :param MI_params: M_params to pass to MultualInformaton_IMSAT module.
        >>> self.criterion = MultualInformaton_IMSAT(**MI_params)
        MI_params can be {mu:4} by default or {mu:16} if the entropy is weak.
        :param VAT_params: VAT_params to pass to VATModuleInterface.
        >>> self.reg_module =  VATModuleInterface(VAT_params)
        VAT_params can be {eps:10, name=kl} or {eps=1, name=mi}. In IMSAT, only name=kl is supported.
        :param kwargs:
        """
        assert VAT_params.get("name", "kl") == "kl", \
            f"In IMSAT framework, KL distance is the only to be supported, given {VAT_params.get('name')}."
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
            MI_params,
            **kwargs,
        )
        self.reg_module = VATModuleInterface(VAT_params)

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        """
        add vat meter
        :return:
        """
        colums = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_adv", AverageValueMeter())
        colums.insert(-1, "train_adv_mean")
        return colums

    @property
    def _training_report_dict(self):
        # add vat meter
        report_dict = super()._training_report_dict
        report_dict.update({"adv": self.METERINTERFACE["train_adv"].summary()["mean"]})
        return dict_filter(report_dict)

    def _regulaze(
            self,
            images: Tensor,
            tf_images: Tensor,
            img_pred_simplex: List[Tensor],
            head_name="B",
    ) -> Tensor:
        """
        return VAT loss for the images and models, in this function, only `images` or `tf1_images` are used for VAT
        :param images: basic transformed image
        :param tf_images: advance transformed image
        :param img_pred_simplex: prediction on basci transformed image
        :param head_name: head_name
        :return: regularization loss
        """
        reg_loss, *_ = self.reg_module(self.model.torchnet, images, head=head_name)
        self.METERINTERFACE["train_adv"].add(reg_loss.item())
        return reg_loss


class IMSATVATGeoTrainer(IMSATVATTrainer):
    """
    This class extends IMSATVATTrainer in order to link the two geometric transformations by a KL divergence
    """

    def __init_meters__(self) -> List[Union[str, List[str]]]:
        columns = super().__init_meters__()
        self.METERINTERFACE.register_new_meter("train_geo", AverageValueMeter())
        return ["train_geo_mean"] + columns

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update({"geo": self.METERINTERFACE["train_geo"].summary()["mean"]})
        return dict_filter(report_dict)

    def _regulaze(
            self,
            images: Tensor,
            tf_images: Tensor,
            img_pred_simplex: List[Tensor],
            head_name="B",
    ) -> Tensor:
        """
        GEO+VAT regularization. VAT is based on `images` and GEO is based on kl(pred_tf_images,image_pred_simplex)
        :param images: basic transformed images, with device = self.device
        :param tf_images: advanced transformed images, with device = self.device
        :param img_pred_simplex: list of simplexes prediction on `images`
        :param head_name: head_name
        :return: regularization tensor to call .backward()
        """
        # VAT loss for images
        vat_loss = super()._regulaze(images, tf_images, img_pred_simplex, head_name)

        # advanced transformed images
        tf_pred_simplex = self.model.torchnet(tf_images, head=head_name)
        assert assert_list(simplex, tf_pred_simplex) and len(tf_pred_simplex) == len(img_pred_simplex)
        # kl div:
        geo_losses: List[Tensor] = []
        for subhead, (tf1_pred, tf2_pred) in enumerate(zip(img_pred_simplex, tf_pred_simplex)):
            assert simplex(tf1_pred) and simplex(tf2_pred)
            geo_losses.append(self.kl_div(tf2_pred, tf1_pred.detach()))
        geo_losses: Tensor = sum(geo_losses) / len(geo_losses)  # type: ignore
        self.METERINTERFACE["train_geo"].add(geo_losses.item())
        # the regularization for the two are 1:1 by default for the sake for simplification.
        return vat_loss + geo_losses


class IMSATMixupTrainer(IMSATVATTrainer):
    """
    implement Mixup in the regularization method
    You will use KL as the distance function to link the two
    No VAT_params can be provided here.
    """

    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = "IMSATAbstractTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,
            config: Dict[str, Union[int, float, str, Dict[str, Any]]] = None,
            MI_params: Dict[str, Union[int, float, str]] = {},
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
            MI_params,
            **kwargs,
        )
        # override the regularzation module
        print("Override VAT module.")
        self.reg_module = MixUp(device=self.device, num_classes=self.model.arch_dict["output_k_B"])
        self.METERINTERFACE.register_new_meter("train_mixup", AverageValueMeter())
        self.drawer.columns_to_draw.insert(-1, "train_mixup_mean")

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update(
            {"mixup": self.METERINTERFACE["train_mixup"].summary()["mean"]}
        )
        # i do not delete the adv meter but I have dict_filter
        return dict_filter(report_dict)

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
        self.METERINTERFACE["train_mixup"].add(reg_losses.item())
        return reg_losses


class IMSATVATMixupTrainer(IMSATVATTrainer):
    """
    this trainer uses VAT+mixup to regularize the clustering
    """

    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = "IMSATVATMixupTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,
            config: Dict[str, Union[int, float, str, Dict[str, Any]]] = None,
            MI_params: Dict[str, Union[int, float, str]] = {},
            VAT_params: Dict[str, Union[int, float, str]] = {},
            **kwargs: Dict[str, Union[int, float, str]],
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
            MI_params,
            VAT_params,
            **kwargs,
        )
        self.mix_module = MixUp(device=self.device, num_classes=self.model.arch_dict["output_k_B"])
        self.METERINTERFACE.register_new_meter("train_mixup", AverageValueMeter())
        self.drawer.columns_to_draw.insert(-1, "train_mixup_mean")

    @property
    def _training_report_dict(self):
        report_dict = super()._training_report_dict
        report_dict.update(
            {"mixup": self.METERINTERFACE["train_mixup"].summary()["mean"]}
        )
        return dict_filter(report_dict)

    def _regulaze(
            self,
            images: Tensor,
            tf_images: Tensor,
            img_pred_simplex: List[Tensor],
            head_name="B",
    ) -> Tensor:
        # here just use the tf1_image to mixup
        # nothing with tf2_images
        vat_loss = super()._regulaze(images, tf_images, img_pred_simplex, head_name)
        mixup_losses: List[Tensor] = []
        for subhead, tf1_pred in enumerate(img_pred_simplex):
            # using image1 and image1.flip(0) to perform the mixup.
            mixup_img, mixup_label, mixup_index = self.mix_module(
                images, tf1_pred, images.flip(0), tf1_pred.flip(0)
            )
            subhead_loss = self.kl_div(self.model(mixup_img)[subhead], mixup_label)
            mixup_losses.append(subhead_loss)
        mixup_losses: Tensor = sum(mixup_losses) / len(mixup_losses)
        self.METERINTERFACE["train_mixup"].add(mixup_losses.item())
        # VAT: mixup = 1:1 for simplification
        return mixup_losses + vat_loss


class IMSATVATGeoMixupTrainer(IMSATVATGeoTrainer):
    def __init__(
            self,
            model: Model,
            train_loader_A: DataLoader,
            train_loader_B: DataLoader,
            val_loader: DataLoader,
            max_epoch: int = 100,
            save_dir: str = "IMSATAbstractTrainer",
            checkpoint_path: str = None,
            device="cpu",
            head_control_params: Dict[str, int] = {"B": 1},
            use_sobel: bool = False,
            config: Dict[str, Union[int, float, str, Dict[str, Any]]] = None,
            MI_params: Dict[str, Union[int, float, str]] = {},
            **kwargs: Dict[str, Union[int, float, str]],
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
            MI_params,
            **kwargs,
        )
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
        # vat geo report_dict
        report_dict = super()._training_report_dict
        report_dict.update(
            {"train_mixup": self.METERINTERFACE["train_mixup"].summary()["mean"]}
        )
        # vat geo mixup report_dict
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
        mixup_losses: List[Tensor] = []
        for subhead, tf1_pred in enumerate(img_pred_simplex):
            mixup_img, mixup_label, mixup_index = self.mixup_module(
                images, tf1_pred, images.flip(0), tf1_pred.flip(0)
            )
            subhead_loss = self.kl_div(self.model(mixup_img)[subhead], mixup_label)
            mixup_losses.append(subhead_loss)
        mixup_loss: Tensor = sum(mixup_losses) / len(mixup_losses)
        self.METERINTERFACE["train_mixup"].add(mixup_loss.item())
        # vat: geo: mixup= 1: 1: 1 for the sake for simplification.
        return mixup_loss + vat_geo_loss
