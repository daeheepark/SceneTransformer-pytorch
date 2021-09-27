import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.functional as FU

import sys, os
import numpy as np
import cv2
import copy
import pytorch_lightning as pl

from model.encoder import Encoder
from model.decoder import Decoder

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

        self.encoder = Encoder(self.device, self.in_feat_dim, self.in_dynamic_rg_dim, self.in_static_rg_dim,
                                    self.time_steps, self.feature_dim, self.head_num)
        self.decoder = Decoder(self.device, self.time_steps, self.feature_dim, self.head_num, self.k, self.F)
        
    def forward(self, states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch,
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
                        agent_rg_mask, agent_traffic_mask):

        encodings = self.encoder(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch,
                                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
                                        agent_rg_mask, agent_traffic_mask)
        decoding = self.decoder(encodings, agents_batch_mask, states_padding_mask_batch)
        
        return decoding.permute(1,2,0,3)

    def training_step(self, batch, batch_idx):

        states_batch, agents_batch_mask, states_padding_mask_batch, \
                (states_hidden_mask_BP, states_hidden_mask_CBP, states_hidden_mask_GDP), \
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch, \
                        agent_rg_mask, agent_traffic_mask = batch

        # TODO : randomly select hidden mask
        states_hidden_mask_batch = states_hidden_mask_BP
        
        no_nonpad_mask = torch.sum((~states_padding_mask_batch*~states_hidden_mask_batch),dim=-1) != 0
        no_nonpad_mask *= ~states_padding_mask_batch[:,self.current_step]

        states_batch = states_batch[no_nonpad_mask]
        agents_batch_mask = agents_batch_mask[no_nonpad_mask][:,no_nonpad_mask]
        states_padding_mask_batch = states_padding_mask_batch[no_nonpad_mask]
        states_hidden_mask_batch = states_hidden_mask_batch[no_nonpad_mask]
        agent_rg_mask = agent_rg_mask[no_nonpad_mask]
        agent_traffic_mask = agent_traffic_mask[no_nonpad_mask]                               
        
        # Predict
        prediction = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
                            agent_rg_mask, agent_traffic_mask)

        # Calculate Loss
        to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch
        gt = states_batch[:,:,:2]
        gt = gt[to_predict_mask]
        prediction = prediction[to_predict_mask]    
        #print(prediction) 
        Loss = nn.MSELoss(reduction='none')
        loss_ = Loss(gt.unsqueeze(1).repeat(1,self.F,1), prediction)
        loss_ = torch.min(torch.mean(torch.mean(loss_, dim=0),dim=-1))
        self.log_dict({'train_loss':loss_})
        return loss_

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
        
        states_batch, agents_batch_mask, states_padding_mask_batch, \
                (states_hidden_mask_BP, states_hidden_mask_CBP, states_hidden_mask_GDP), \
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch, \
                        agent_rg_mask, agent_traffic_mask = batch

        # TODO : randomly select hidden mask
        states_hidden_mask_batch = states_hidden_mask_BP
        
        no_nonpad_mask = torch.sum((~states_padding_mask_batch*~states_hidden_mask_batch),dim=-1) != 0
        no_nonpad_mask *= ~states_padding_mask_batch[:,self.current_step]

        states_batch = states_batch[no_nonpad_mask]
        agents_batch_mask = agents_batch_mask[no_nonpad_mask][:,no_nonpad_mask]
        states_padding_mask_batch = states_padding_mask_batch[no_nonpad_mask]
        states_hidden_mask_batch = states_hidden_mask_batch[no_nonpad_mask]
        agent_rg_mask = agent_rg_mask[no_nonpad_mask]
        agent_traffic_mask = agent_traffic_mask[no_nonpad_mask]    
        
        # Predict
        prediction = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
                            agent_rg_mask, agent_traffic_mask)

        # Calculate Loss
        to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch
        
        gt = states_batch[:,:,:2]
        # gt = gt[to_predict_mask]                                                # [A*T,2]
        # prediction = prediction[to_predict_mask]                                # [A*T,candi,2]
        
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

        self.log_dict({'val_loss': loss_, 'minade': batch_minade, 'minfde': batch_minfde, 'avgade': batch_avgade, 'avgfde': batch_avgfde})

    def test_step(self, batch, batch_idx):
        
        states_batch, agents_batch_mask, states_padding_mask_batch, \
                (states_hidden_mask_BP, states_hidden_mask_CBP, states_hidden_mask_GDP), \
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch, \
                        agent_rg_mask, agent_traffic_mask = batch

        # TODO : randomly select hidden mask
        states_hidden_mask_batch = states_hidden_mask_BP
        
        no_nonpad_mask = torch.sum((~states_padding_mask_batch*~states_hidden_mask_batch),dim=-1) != 0
        no_nonpad_mask *= ~states_padding_mask_batch[:,self.current_step]

        states_batch = states_batch[no_nonpad_mask]
        agents_batch_mask = agents_batch_mask[no_nonpad_mask][:,no_nonpad_mask]
        states_padding_mask_batch = states_padding_mask_batch[no_nonpad_mask]
        states_hidden_mask_batch = states_hidden_mask_batch[no_nonpad_mask]
        agent_rg_mask = agent_rg_mask[no_nonpad_mask]
        agent_traffic_mask = agent_traffic_mask[no_nonpad_mask]    
        
        # Predict
        prediction = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
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

        self.log_dict({'val_loss': loss_, 'minade': batch_minade, 'minfde': batch_minfde, 'avgade': batch_avgade, 'avgfde': batch_avgfde})


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9,0.999))
        return optimizer

def xy_to_pixel(xy, width):
    xy[:,1] *= -1
    xy += int(width/2)
    mask_x = (xy[:,0] <= width)*(xy[:,0]>=0)
    mask_y = (xy[:,1] <= width)*(xy[:,1]>=0)

    xy = xy[mask_x*mask_y]

    return xy