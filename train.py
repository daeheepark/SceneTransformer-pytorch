import os
import sys
import hydra

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn
from model.pl_module import SceneTransformer

@hydra.main(config_path='./conf', config_name='config.yaml')
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_ids

    pl.seed_everything(cfg.seed)
    pwd = hydra.utils.get_original_cwd() + '/'
    print('Current Path: ', pwd)

    dataset_train = WaymoDataset(pwd+cfg.dataset.train.tfrecords, pwd+cfg.dataset.train.idxs)
    dataset_valid = WaymoDataset(pwd+cfg.dataset.valid.tfrecords, pwd+cfg.dataset.valid.idxs)    
    dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.train.batchsize, collate_fn=waymo_collate_fn, num_workers=16)
    dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.valid.batchsize, collate_fn=waymo_collate_fn, num_workers=16)

    model = SceneTransformer(cfg)

    trainer = pl.Trainer(max_epochs=cfg.max_epochs, gpus=cfg.device_num, gradient_clip_val=5,accelerator='ddp',val_check_interval=4000, 
                                limit_train_batches=1.0, limit_val_batches=0.4, log_every_n_steps=100)
    # trainer = pl.Trainer(max_epochs=cfg.max_epochs, gpus=cfg.device_num, gradient_clip_val=5,accelerator='ddp')
    trainer.fit(model, dloader_train, dloader_valid)


if __name__ == '__main__':
    sys.exit(main())
