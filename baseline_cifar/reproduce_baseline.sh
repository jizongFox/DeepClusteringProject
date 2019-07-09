#!/usr/bin/env bash
python train_original.py Trainer.name=imsat Trainer.save_dir=cifar_paper_baseline/imsat
python train_original.py Trainer.name=imsatvat Trainer.save_dir=cifar_paper_baseline/imsatvat