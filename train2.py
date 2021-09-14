import os, glob
import sys
import hydra

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datautil.waymo_tfrecord_dataset import WaymoTFDataset, waymo_collate_fn
from model.pl_module import SceneTransformer

import tensorflow as tf


@hydra.main(config_path='./conf', config_name='config.yaml')
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_ids

    pl.seed_everything(cfg.seed)
    pwd = hydra.utils.get_original_cwd() + '/'
    print('Current Path: ', pwd)

    fns_train = glob.glob(pwd+cfg.dataset.train.tfrecords+'/*')
    fns_valid = glob.glob(pwd+cfg.dataset.valid.tfrecords+'/*')
    dataset_train = WaymoTFDataset(fns_train)
    dataset_valid = WaymoTFDataset(fns_valid)
    print('dataset loaded')
    dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.train.batchsize, collate_fn=waymo_collate_fn)
    dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.valid.batchsize, collate_fn=waymo_collate_fn)

    model = SceneTransformer(cfg)

    # trainer = pl.Trainer(max_epochs=cfg.max_epochs, gpus=cfg.device_num, accelerator='ddp',val_check_interval=2000, 
    #                             limit_train_batches=0.5, limit_val_batches=0.5, log_every_n_steps=1000)
    trainer = pl.Trainer(max_epochs=cfg.max_epochs, gpus=cfg.device_num, gradient_clip_val=5,accelerator='ddp')
    trainer.fit(model, dloader_train, dloader_valid)

if __name__ == '__main__':
    sys.exit(main())
