import os, shutil
import sys
import hydra
from pytorch_lightning import callbacks

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn
from model.pl_module import SceneTransformer

@hydra.main(config_path='./conf', config_name='config.yaml')
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_ids

    pl.seed_everything(cfg.seed)
    pwd = hydra.utils.get_original_cwd() + '/'
    print('Current Path: ', pwd)

    dataset_train = WaymoDataset(pwd+cfg.dataset.train.tfrecords, pwd+cfg.dataset.train.idxs, shuffle_queue_size=cfg.dataset.train.batchsize)
    dataset_valid = WaymoDataset(pwd+cfg.dataset.valid.tfrecords, pwd+cfg.dataset.valid.idxs, shuffle_queue_size=None)    
    dataset_test = WaymoDataset(pwd+cfg.dataset.test.tfrecords, pwd+cfg.dataset.test.idxs, shuffle_queue_size=None)
    dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.train.batchsize, collate_fn=waymo_collate_fn, num_workers=cfg.dataset.train.batchsize)
    dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.valid.batchsize, collate_fn=waymo_collate_fn, num_workers=cfg.dataset.valid.batchsize)
    dloader_test = DataLoader(dataset_test, batch_size=cfg.dataset.test.batchsize, collate_fn=waymo_collate_fn, num_workers=cfg.dataset.test.batchsize)

    model = SceneTransformer(cfg)
    checkpoint_callback = ModelCheckpoint(mode='min', monitor='val_loss', auto_insert_metric_name=True, verbose=True, save_last=True, 
                                            save_top_k=3)
    checkpoint_callback_2 = ModelCheckpoint(mode='min', monitor='minfde', auto_insert_metric_name=True, verbose=True, save_last=True, 
                                            save_top_k=3)

    trainer = pl.Trainer(max_epochs=cfg.max_epochs, gpus=cfg.device_num, accelerator='ddp', val_check_interval=0.2,
                                limit_train_batches=1.0, limit_val_batches=1.0, log_every_n_steps=100, callbacks=[checkpoint_callback,checkpoint_callback_2])

    if cfg.mode == 'train':
        trainer.fit(model, dloader_train, dloader_valid)
    elif cfg.mode == 'validate':
        trainer.validate(model, dataloaders=dloader_valid, verbose=True)
    elif cfg.mode == 'test':
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
