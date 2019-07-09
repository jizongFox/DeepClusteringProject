#!/usr/bin/env bash
python train_original.py Trainer.name=imsat Trainer.save_dir=cifar_paper_baseline/imsat_01
python train_original.py Trainer.name=imsatvat Trainer.save_dir=cifar_paper_baseline/imsatvat_02 Trainer.max_epoch=200 Trainer.VAT_params={eps:1}
