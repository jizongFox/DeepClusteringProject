from typing import Type

from .clustering_trainer import *
from .iic_regularized_trainer import *
from .iic_trainer import *
from .imsat_trainer import *

trainer_mapping: Dict[str, Type[ClusteringGeneralTrainer]] = {
    # using different transforms for iic
    "iicgeo": IICGeoTrainer,  # the basic iic
    "iicmixup": IICMixupTrainer,  # the basic IIC with mixup as the data augmentation
    "iicvat": IICVATTrainer,  # the basic iic with VAT as the basic data augmentation
    "iicgaussian": IICGaussianTrainer,  # the basic iic with gaussian data augmentation todo:check
    "iiccutout": IICCutoutTrainer,  # the basic iic with cutout as data augmentation todo:check
    "iicgeovat": IICGeoVATTrainer,  # IIC with geo and vat as the data augmentation
    "iicgeomixup": IICGeoMixupTrainer,  # IIC with geo and mixup as the data augmentation

    "iicgeogaussian": IICGeoGaussianTrainer,  # IIC with geo and gaussian as the data augmentation todo: check
    "iicgeocutout": IICGeoCutoutTrainer,  # IIC with geo and cutout as the data augmentation todo: check
    "iicgeovatmixup": IICGeoVATMixupTrainer,  # IIC with geo, vat and mixup as the data augmentation

    # using different regularization for imsat
    "imsat": IMSATAbstractTrainer,  # imsat without any regularization
    "imsatvat": IMSATVATTrainer,  # imsat with vat
    "imsatgeo": IMSATGeoTrainer,  # imsat with geo transformation
    "imsatmixup": IMSATMixupTrainer,  # imsat with mixup
    "imsatgaussian": IMSATGaussianTrainer,  # imsat with gaussian noise # todo: checkout
    "imsatcutout": IMSATCutoutTrainer,  # imsat with cutout transform # todo: checkout
    "imsatvatmixup": IMSATVATMixupTrainer,  # imsat with vat + mixup
    "imsatvatgeo": IMSATVATGeoTrainer,  # imsat with geo+vat
    "imsatgeomixup": IMSATGeoMixupTrainer,  # imsat with geo and mixup
    "imsatvatgeomixup": IMSATVATGeoMixupTrainer,  # imsat with geo vat and mixup
    "imsatvatiicgeo": IMSATVATIICGeoTrainer,  # using IMSATVAT with IIC regularization # todo: checkout
    "imsatvatcutout": IMSATVATCutoutTrainer,  # todo: checkout
    "imsatmixupcutout": IMSATMixupCutoutTrainer,  # todo: checkout
    "imsatgeovatgutoutgaussian": IMSATGeoVATCutoutGaussianTrainer,


    # using different regularization for iic
    "iicgeovatreg": IICVAT_RegTrainer,  # iicgeo with VAT as a regularization
    "iicgeomixupreg": IICMixup_RegTrainer,  # iicgeo with mixup as a regularization
    "iicgeovatmixupreg": IICVATMixup_RegTrainer,  # iicgeo with VAT and mixup as a regularization
    "iicgeovatvatreg": IICVATVAT_RegTrainer,  # iicgeo with VAT and VAT as regularization
    "iicvatmivatklreg": IICVATMI_VATKLTrainer,  # special case of IIC with regularization,
    # using VAT_mi for IIC and VAT_kl for Regularization
    "iicgeocutoutreg": IICCutout_RegTrainer,  # todo: check
    "iicgeogaussianreg": IICGaussian_RegTrainer,  # todo: checkout
    "iicgeovatcutoutreg": IICVATCutout_RegTrainer,  # todo:checkout
    "iicgeovatgaussianreg": IICVATGaussian_RegTrainer,  # todo:checkout
    "iicgeovatcutoutgaussianreg": IICVATCutoutGaussian_RegTrainer,  # todo:checkout
    "iicgeovatmixupcutoutreg": IICVATMixupCutout_RegTrainer,  # todo:checkout
    "iicgeomixupgaussianreg": IICMixupGaussian_RegTrainer,  # todo:checkout

}
