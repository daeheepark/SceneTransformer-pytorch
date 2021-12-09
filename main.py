import os
import os.path as osp
import shutil
import sys
import hydra
import logging
from pytorch_lightning import callbacks

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn, waymo_worker_fn
from model.pl_module import SceneTransformer

@hydra.main(config_path='./conf', config_name='config.yaml')
def main(cfg):
    pl.seed_everything(cfg.seed)

    logger = logging.getLogger("pytorch_lightning")
    logger.addHandler(logging.FileHandler("terminal.log"))
    
    pwd = hydra.utils.get_original_cwd()
    print('Current Path: ', pwd)


    model = SceneTransformer(cfg)
    checkpoint_callback = ModelCheckpoint(mode='min', monitor='val/loss', auto_insert_metric_name=True, verbose=True, save_last=True, save_top_k=3)
    checkpoint_callback_2 = ModelCheckpoint(mode='min', monitor='val/minfde', auto_insert_metric_name=True, verbose=True, save_last=True, save_top_k=3)
    early_stopping = callbacks.EarlyStopping(monitor='val/loss', mode='min', patience=5)

    trainer_args = {'max_epochs': cfg.max_epochs,
                    'gpus': cfg.gpu_ids,
                    'accelerator': 'ddp',
                    'val_check_interval': cfg.dataset.train.val_check_interval, 'limit_train_batches': cfg.dataset.train.limit_train_batches, 
                    'limit_val_batches': cfg.dataset.train.limit_val_batches,
                    'log_every_n_steps': cfg.max_epochs, 'auto_lr_find': True,
                    'callbacks': [checkpoint_callback, checkpoint_callback_2, early_stopping],
                    'resume_from_checkpoint': cfg.resume}

    trainer = pl.Trainer(**trainer_args)

    if cfg.mode == 'train':
        dataset_train = WaymoDataset(osp.join(pwd, cfg.dataset.train.tfrecords), osp.join(pwd, cfg.dataset.train.idxs), shuffle_queue_size=cfg.dataset.train.batchsize)
        dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.train.batchsize, 
                                    collate_fn=lambda b: waymo_collate_fn(b, halfwidth=cfg.dataset.halfwidth, only_veh=cfg.dataset.only_veh, hidden=cfg.dataset.hidden),
                                    worker_init_fn=waymo_worker_fn,  
                                    num_workers=cfg.dataset.valid.batchsize)

        dataset_valid = WaymoDataset(osp.join(pwd, cfg.dataset.valid.tfrecords), osp.join(pwd, cfg.dataset.valid.idxs), shuffle_queue_size=None)
        dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.valid.batchsize, 
                                    collate_fn=lambda b: waymo_collate_fn(b, halfwidth=cfg.dataset.halfwidth, only_veh=cfg.dataset.only_veh, hidden=cfg.dataset.hidden),
                                    worker_init_fn=waymo_worker_fn,  
                                    num_workers=cfg.dataset.valid.batchsize)

        trainer.fit(model, dloader_train, dloader_valid)
    elif cfg.mode == 'validate':
        dataset_valid = WaymoDataset(osp.join(pwd, cfg.dataset.valid.tfrecords), osp.join(pwd, cfg.dataset.valid.idxs), shuffle_queue_size=None)
        dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.valid.batchsize, 
                                    collate_fn=lambda b: waymo_collate_fn(b, halfwidth=cfg.dataset.halfwidth, only_veh=cfg.dataset.only_veh, hidden=cfg.dataset.hidden),
                                    worker_init_fn=waymo_worker_fn,  
                                    num_workers=cfg.dataset.valid.batchsize)

        trainer.validate(model, dataloaders=dloader_valid, verbose=True)
    elif cfg.mode == 'test':
        dataset_test = WaymoDataset(osp.join(pwd, cfg.dataset.test.tfrecords), osp.join(pwd, cfg.dataset.test.idxs), shuffle_queue_size=None)
        dloader_test = DataLoader(dataset_test, batch_size=cfg.dataset.test.batchsize, collate_fn=waymo_collate_fn, worker_init_fn=waymo_worker_fn, num_workers=cfg.dataset.test.batchsize)

        dir = os.path.join(os.getcwd(),'results')
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        model = model.load_from_checkpoint(checkpoint_path=cfg.dataset.test.ckpt_path, cfg=cfg)
        trainer.test(model, dataloaders=dloader_test, verbose=True)
    else:
        raise KeyError


if __name__ == '__main__':
    sys.exit(main())
