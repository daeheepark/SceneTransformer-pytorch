import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.functional as FU
from torchvision import transforms

import sys, os
import os.path as osp
import numpy as np
import cv2
import copy
import hydra
import pytorch_lightning as pl

sys.path.append('/home/user/daehee/SceneTransformer-pytorch')

from model.encoder import Encoder
from model.decoder import Decoder
from datautil.waymo_dataset import xy_to_pixel

COLORS = [(0,0,255), (255,0,255), (180,180,0), (143,143,188), (0,100,0), (128,128,0)]

class SceneTransformer(pl.LightningModule):
    def __init__(self, cfg):
        super(SceneTransformer, self).__init__()
        self.cfg = cfg
        self.in_feat_dim = cfg.model.in_feat_dim
        self.in_dynamic_rg_dim = cfg.model.in_dynamic_rg_dim
        self.in_static_rg_dim = cfg.model.in_static_rg_dim
        self.time_steps = cfg.model.time_steps
        self.current_step = cfg.model.current_step
        self.feature_dim = cfg.model.feature_dim
        self.head_num = cfg.model.head_num
        self.k = cfg.model.k
        self.F = cfg.model.F

        self.Loss = nn.MSELoss(reduction='none')

        self.encoder = Encoder(self.device, self.in_feat_dim, self.in_dynamic_rg_dim, self.in_static_rg_dim,
                                    self.time_steps, self.feature_dim, self.head_num)
        self.decoder = Decoder(self.device, self.time_steps, self.feature_dim, self.head_num, self.k, self.F)

        ### viz options
        self.width = cfg.viz.width
        self.totensor = transforms.ToTensor()
        
    def forward(self, states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch,
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                        agent_rg_mask, agent_traffic_mask):

        encodings = self.encoder(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch,
                                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                                        agent_rg_mask, agent_traffic_mask)
        # states_padding_mask_batch = states_padding_mask_batch + ~states_hidden_mask_batch
        decoding = self.decoder(encodings, agents_batch_mask, states_padding_mask_batch)
        
        return decoding.permute(1,2,0,3)

    def training_step(self, batch, batch_idx):

        states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, \
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch, \
                        agent_rg_mask, agent_traffic_mask, (num_agents_accum, num_rg_accum, num_tl_accum), \
                            sdc_masks, center_ps = batch.values()
        
        # Predict
        prediction = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                            agent_rg_mask, agent_traffic_mask)

        # Calculate Loss
        to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch
        gt = states_batch[:,:,:2]
        gt = gt[to_predict_mask]
        prediction = prediction[to_predict_mask]    
        #print(prediction) 
        loss_ = self.Loss(gt.unsqueeze(1).repeat(1,self.F,1), prediction)
        loss_ = torch.min(torch.mean(torch.mean(loss_, dim=0),dim=-1)) * self.cfg.dataset.halfwidth
        self.log_dict({'train/loss':loss_})
        
        return {'batch': batch, 'pred': prediction, 'gt': gt, 'loss': loss_}

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def validation_step(self, batch, batch_idx):
        
        states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, \
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch, \
                        agent_rg_mask, agent_traffic_mask, (num_agents_accum, num_rg_accum, num_tl_accum), \
                            sdc_masks, center_ps = batch.values()
        
        # Predict
        prediction = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                            agent_rg_mask, agent_traffic_mask)

        # Calculate Loss
        to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch
        
        gt = states_batch[:,:,:2]
        # gt = gt[to_predict_mask]                                                # [A*T,2]
        # prediction = prediction[to_predict_mask]                                # [A*T,candi,2]
        
        loss_ = self.Loss(gt[to_predict_mask].unsqueeze(1).repeat(1,self.F,1), prediction[to_predict_mask])            # [A*T,candi]
        loss_ = torch.min(torch.mean(torch.mean(loss_, dim=0),dim=-1)) * self.cfg.dataset.halfwidth

        rs_error = ((prediction - gt.unsqueeze(2)) ** 2).sum(dim=-1).sqrt_() * self.cfg.dataset.halfwidth
        rs_error[~to_predict_mask]=0
        rse_sum = rs_error.sum(1)
        ade_mask = to_predict_mask.sum(-1)!=0
        ade = (rse_sum[ade_mask].permute(1,0)/to_predict_mask[ade_mask].sum(-1)).permute(1,0)

        fde_mask = to_predict_mask[:,-1]==True
        fde = rs_error[fde_mask][:,-1,:]

        minade, _ = ade.min(dim=-1)
        avgade = ade.mean(dim=-1)
        minfde, _ = fde.min(dim=-1)
        avgfde = fde.mean(dim=-1)

        batch_minade = minade.mean()
        batch_minfde = minfde.mean()
        batch_avgade = avgade.mean()
        batch_avgfde = avgfde.mean()

        self.log_dict({'val/loss': loss_, 'val/minade': batch_minade, 'val/minfde': batch_minfde, 'val/avgade': batch_avgade, 'val/avgfde': batch_avgfde})

        return {'states': states_batch, 'states_padding': states_padding_mask_batch, 'states_hidden': states_hidden_mask_batch, 
                'roadgraph_feat': roadgraph_feat_batch, 'roadgraph_padding': roadgraph_padding_batch, 
                'traffic_light_feat': traffic_light_feat_batch, 'traffic_light_padding': traffic_light_padding_batch,
                'num_agents_accum': num_agents_accum, 'num_rg_accum': num_rg_accum, 'num_tl_accum': num_tl_accum, 'center_ps': center_ps,
                'pred': prediction, 'loss': loss_}

    def validation_epoch_end(self, outputs) -> None:
        states_batch, states_padding_batch, states_hidden_batch, \
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch, \
                        num_agents_accum, num_rg_accum, num_tl_accum, center_ps, \
                            prediction, loss = outputs[0].values()

        empty_mask = np.zeros((self.width,self.width))
        total_empty = np.zeros((self.width,self.width,3))
        current_step=3
        scene_imgs = []

        for i in range(len(num_agents_accum)-1):
            total_empty_ = total_empty.copy()
            start = num_agents_accum[i]
            end = num_agents_accum[i+1]
            states_, states_padding_, states_hidden_ = states_batch[start:end].cpu(), states_padding_batch[start:end].cpu(), states_hidden_batch[start:end].cpu()
            pred_ = prediction[start:end].detach().cpu()
            # roadgraph_feat_, roadgraph_padding_ = roadgraph_feat_batch[1400*i:1400*(i+1)].cpu(), roadgraph_padding_batch[1400*i:1400*(i+1)].cpu()
            roadgraph_feat_, roadgraph_padding_ = roadgraph_feat_batch[num_rg_accum[i]:num_rg_accum[i+1]].cpu(), roadgraph_padding_batch[num_rg_accum[i]:num_rg_accum[i+1]].cpu()
            
            roadgraph_type_, roadgraph_id_ = roadgraph_feat_[:,current_step,-2].cpu(), roadgraph_feat_[:,current_step,-1].cpu()
            traffic_light_, traffic_light_padding_ = traffic_light_feat_batch[num_tl_accum[i]:num_tl_accum[i+1]].cpu(), traffic_light_padding_batch[num_tl_accum[i]:num_tl_accum[i+1]].cpu()
            center_p = center_ps[i].cpu()

            ctline_mask = (roadgraph_feat_[...,-2]==2)[:,0]

            # # Road 
            # lane_mask = torch.zeros(ctline_mask.shape, dtype=torch.bool)
            # for type_ in [6,7,8,9,10,11,12,13,14]:
            #     lane_mask += (roadgraph_type_==type_)

            # for id_ in np.unique(roadgraph_id_[lane_mask]):
            #     lane_id_mask = lane_mask*(roadgraph_id_==id_)
            #     lane_xy = roadgraph_feat_[:,current_step,:2][lane_id_mask] - center_p
            #     lane_xy *= (self.width/2)
            #     lane_xy = xy_to_pixel(lane_xy,self.width)
            #     polygon = np.array([lane_xy.numpy()], np.int32)
            #     cv2.polylines(total_empty_, polygon, isClosed=False, color=(125,125,125), thickness=20)

            for id_ in np.unique(roadgraph_id_[ctline_mask]):
                ctline_id_mask = ctline_mask*(roadgraph_id_==id_)
                ctline_xy = roadgraph_feat_[:,current_step,:2][ctline_id_mask] - center_p
                ctline_xy *= (self.width/2)
                ctline_xy = xy_to_pixel(ctline_xy,self.width)

                polygon = np.array([ctline_xy.numpy()], np.int32)

                cv2.polylines(total_empty_, polygon, isClosed=False, color=(255,255,255), thickness=20)

            for si_, (s_, p_, sp_, sh_) in enumerate(zip(states_, pred_, states_padding_, states_hidden_)):
                s__ = s_[:,:2].clone() * (self.width/2)
                s__ = xy_to_pixel(s__, self.width)
                p__ = p_.clone() * (self.width/2)
                p__ = xy_to_pixel(p__[sh_], self.width)
                curpt_ = s__[current_step].unsqueeze(0).repeat(6,1).unsqueeze(0)
                p__ = torch.cat((curpt_, p__), 0).permute(1,0,2)

                if not sp_[current_step]:
                    polygon = np.array([s__[~sp_].numpy()], np.int32)
                    cv2.polylines(total_empty_, polygon, isClosed=False, color=COLORS[si_%len(COLORS)], thickness=2)
                    cv2.circle(img=total_empty_, center=tuple(s__[current_step].numpy().astype(np.int32)), radius=3, color=COLORS[si_%len(COLORS)], thickness=cv2.FILLED)

                    for p___ in p__:
                        mask_x = (p___[...,0] <= self.width)*(p___[...,0]>=0)
                        mask_y = (p___[...,1] <= self.width)*(p___[...,1]>=0)
                        p___ = p___[mask_x*mask_y]
                        # polygon.append(p___.numpy())
                        polygon = np.array([p___.numpy()], np.int32)
                        cv2.polylines(total_empty_, polygon, isClosed=False, color=(200,0,0), thickness=1)

                else:
                    polygon = np.array([s__[~sp_].numpy()], np.int32)
                    cv2.polylines(total_empty_, polygon, isClosed=False, color=(150,150,150), thickness=2)

            red_mask = ((traffic_light_[:,current_step,-1] == 4) + (traffic_light_[:,current_step,-1] == 7))
            yellow_mask = ((traffic_light_[:,current_step,-1] == 5) + (traffic_light_[:,current_step,-1] == 8))
            green_mask = traffic_light_[:,current_step,-1] == 6

            red_xy = xy_to_pixel(traffic_light_[:,current_step,:2][red_mask].clone() - center_p, self.width).numpy()#*(width/2)
            yellow_xy = xy_to_pixel(traffic_light_[:,current_step,:2][yellow_mask].clone() - center_p, self.width).numpy()#*(width/2)
            green_xy = xy_to_pixel(traffic_light_[:,current_step,:2][green_mask].clone() - center_p, self.width).numpy()#*(width/2)

            [cv2.circle(img=total_empty_, center=tuple(xy.astype(np.int32)), radius=5, color=(0,0,255), thickness=1) for xy in red_xy]
            [cv2.circle(img=total_empty_, center=tuple(xy.astype(np.int32)), radius=5, color=(255,255,0), thickness=1) for xy in yellow_xy]
            [cv2.circle(img=total_empty_, center=tuple(xy.astype(np.int32)), radius=5, color=(0,255,0), thickness=1) for xy in green_xy]

            total_empty_ = cv2.cvtColor(total_empty_.astype(np.uint8), cv2.COLOR_BGR2RGB)

            scene_imgs.append((self.totensor(total_empty_)).unsqueeze(0))
        scene_imgs = torch.cat(scene_imgs, dim=0)
        self.logger.experiment.add_image('val/viz', scene_imgs, self.global_step, dataformats="NCHW")

        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, \
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch, \
                        agent_rg_mask, agent_traffic_mask, (num_agents_accum, num_rg_accum, num_tl_accum), \
                            sdc_masks, center_ps = batch.values()
        
        # Predict
        prediction = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                            agent_rg_mask, agent_traffic_mask)

        # Calculate Loss
        to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch
        gt = states_batch[:,:,:2]


        # Visualize
        width = 1000
        empty_pane = np.ones((width,width,3))*255
        roadgraph_type = roadgraph_feat_batch[:,0,-2].detach().cpu().type(torch.int32).numpy()
        roadgraph_id = roadgraph_feat_batch[:,0,-1].detach().cpu().type(torch.int32).numpy()
        mask_ctline = (roadgraph_type==2)
        mask_redge = (roadgraph_type==15)

        for id_ in np.unique(roadgraph_id[mask_ctline]):
            ctline_id_mask = mask_ctline*(roadgraph_id==id_)
            xy_ctline = roadgraph_feat_batch[:,0,:2][ctline_id_mask]
            xy_ctline *= int(width/2)
            xy_ctline[:,1] *= -1
            xy_ctline += int(width/2)
            mask_x = (xy_ctline[:,0] <= width)*(xy_ctline[:,0]>=0)
            mask_y = (xy_ctline[:,1] <= width)*(xy_ctline[:,1]>=0)

            xy_ctline = xy_ctline[mask_x*mask_y]

            polygon = np.array([xy_ctline.detach().cpu().numpy()], np.int32)
            cv2.polylines(empty_pane, polygon, isClosed=False, color=(0,0,0), thickness=1)
        
        gt_ = copy.deepcopy(gt)
        gt_ = gt_.detach().cpu().numpy()

        paddings_ = copy.deepcopy(states_padding_mask_batch)
        paddings_ = paddings_.detach().cpu().numpy()

        prediction_ = copy.deepcopy(prediction)
        prediction_ = prediction_.detach().cpu().numpy()

        predmask_ = copy.deepcopy(to_predict_mask)
        predmask_ = predmask_.detach().cpu().numpy()

        for i, xy_ in enumerate(gt_):
            xy_ *= int(width/2)
            xy_[:,1] *= -1
            xy_ += int(width/2)
            
            xy_past_, xy_future_ = xy_[:self.cfg.model.current_step+1,:], xy_[self.cfg.model.current_step+1:,:]
            xy_past_padding = paddings_[i,:self.cfg.model.current_step+1]
            xy_future_padding = paddings_[i,self.cfg.model.current_step+1:]
            
            xy_past_, xy_future_ = xy_past_[~xy_past_padding], xy_future_[~xy_future_padding]

            polygon = np.array([xy_past_], np.int32)
            cv2.polylines(empty_pane, polygon, isClosed=False, color=(255,0,0), thickness=2)
            polygon = np.array([xy_future_], np.int32)
            cv2.polylines(empty_pane, polygon, isClosed=False, color=(0,255,0), thickness=2)

            # draw prediction
            pred_ = prediction_[i]
            pmask_ = predmask_[i]
            num_candi = pred_.shape[1]
            for ci in range(num_candi):
                predxy_ = copy.deepcopy(pred_[:,ci,:])
                predxy_ = predxy_[pmask_]

                predxy_ *= int(width/2)
                predxy_[:,1] *= -1
                predxy_ += int(width/2)
                polygon = np.array([predxy_], np.int32)
                cv2.polylines(empty_pane, polygon, isClosed=False, color=COLORS[ci], thickness=1)

        imgnm = os.path.join(os.getcwd(),'results',f'{batch_idx}_{i}.jpg')
        cv2.imwrite(imgnm, empty_pane)
        
        Loss = nn.MSELoss(reduction='none')
        loss_ = Loss(gt[to_predict_mask].unsqueeze(1).repeat(1,self.F,1), prediction[to_predict_mask])            # [A*T,candi]
        loss_ = torch.min(torch.mean(torch.mean(loss_, dim=0),dim=-1))  

        rs_error = ((prediction - gt.unsqueeze(2)) ** 2).sum(dim=-1).sqrt_()
        rs_error[~to_predict_mask]=0
        rse_sum = rs_error.sum(1)
        ade_mask = to_predict_mask.sum(-1)!=0
        ade = (rse_sum[ade_mask].permute(1,0)/to_predict_mask[ade_mask].sum(-1)).permute(1,0)

        fde_mask = to_predict_mask[:,-1]==True
        fde = rs_error[fde_mask][:,-1,:]

        minade, _ = ade.min(dim=-1)
        avgade = ade.mean(dim=-1)
        minfde, _ = fde.min(dim=-1)
        avgfde = fde.mean(dim=-1)

        batch_minade = minade.mean()
        batch_minfde = minfde.mean()
        batch_avgade = avgade.mean()
        batch_avgfde = avgfde.mean()

        self.log_dict({'test/loss': loss_, 'test/minade': batch_minade, 'test/minfde': batch_minfde, 'test/avgade': batch_avgade, 'test/avgfde': batch_avgfde})


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9,0.999))
        return optimizer

@hydra.main(config_path='../conf', config_name='config.yaml')
def test_valend(cfg):
    from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn
    from model.pl_module import SceneTransformer

    # trainer_args = {}
    trainer_args = {'max_epochs': cfg.max_epochs,
                    'gpus': [2,3],#cfg.gpu_ids,
                    'accelerator': 'ddp',
                    'val_check_interval': 0.2, 'limit_train_batches': 1.0, 
                    'limit_val_batches': 0.001,
                    'log_every_n_steps': 100, 'auto_lr_find': True,
                    }

    trainer = pl.Trainer(**trainer_args)
    model = SceneTransformer(cfg)
    model = model.load_from_checkpoint(checkpoint_path=cfg.dataset.test.ckpt_path, cfg=cfg)

    pwd = hydra.utils.get_original_cwd()
    dataset_valid = WaymoDataset(osp.join(pwd, cfg.dataset.valid.tfrecords), osp.join(pwd, cfg.dataset.valid.idxs), shuffle_queue_size=None)
    dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.valid.batchsize, shuffle=False, collate_fn=waymo_collate_fn, num_workers=cfg.dataset.valid.batchsize)

    trainer.validate(model=model, val_dataloaders=dloader_valid, verbose=True)

if __name__ == '__main__':
    test_valend()