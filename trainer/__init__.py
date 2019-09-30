from typing import Type

from .clustering_trainer import *
from .iic_regularized_trainer import *
from .iic_trainer import *
from .imsat_trainer import *

trainer_mapping: Dict[str, Type[ClusteringGeneralTrainer]] = {
    # using different transforms for iic
    # ToDo: add cutout or gaussian if it is necessary
    "iicgeo": IICGeoTrainer,  # the basic iic
    "iicmixup": IICMixupTrainer,  # the basic IIC with mixup as the data augmentation
    "iicvat": IICVATTrainer,  # the basic iic with VAT as the basic data augmentation
    "iicgaussian": IICGaussianTrainer,  # the basic iic with gaussian data augmentation
    "iiccutout": IICCutoutTrainer,  # the basic iic with cutout as data augmentation
    "iicgeovat": IICGeoVATTrainer,  # IIC with geo and vat as the data augmentation
    "iicgeomixup": IICGeoMixupTrainer,  # IIC with geo and mixup as the data augmentation
    "iicgeovatmixup": IICGeoVATMixupTrainer,  # IIC with geo, vat and mixup as the data augmentation
    # using different regularization for iic
    "iicgeovatreg": IICVAT_RegTrainer,  # iicgeo with VAT as a regularization
    "iicgeomixupreg": IICMixup_RegTrainer,  # iicgeo with mixup as a regularization
    "iicgeovatmixupreg": IICVATMixup_RegTrainer,  # iicgeo with VAT and mixup as a regularization
    "iicgeovatvatreg": IICVATVAT_RegTrainer,  # iicgeo with VAT and VAT as regularization
    "iicvatmivatklreg": IICVATMI_VATKLTrainer,  # special case of IIC with regularization,
    # using VAT_mi for IIC and VAT_kl for Regularization

    # using different regularization for imsat
    "imsat": IMSATAbstractTrainer,  # imsat without any regularization
    "imsatvat": IMSATVATTrainer,  # imsat with vat
    "imsatgeo": IMSATGeoTrainer,  # imsat with geo transformation
    "imsatmixup": IMSATMixupTrainer,  # imsat with mixup
    "imsatgaussian": IMSATGaussianTrainer,  # imsat with gaussian noise
    "imsatcutout": IMSATCutoutTrainer,  # imsat with cutout transform
    "imsatvatmixup": IMSATVATMixupTrainer,  # imsat with vat + mixup
    "imsatvatgeo": IMSATVATGeoTrainer,  # imsat with geo+vat
    "imsatgeomixup": IMSATGeoMixup,  # imsat with geo and mixup
    "imsatvatgeomixup": IMSATVATGeoMixupTrainer,  # imsat with geo vat and mixup
    "imsatvatiicgeo": IMSATVATIICGeo  # using IMSATVAT with IIC regularization
}
