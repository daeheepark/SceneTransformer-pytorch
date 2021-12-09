import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.functional as FU

import sys
import pytorch_lightning as pl

from model.encoder import Encoder
from model.decoder import Decoder

class SceneTransformer(pl.LightningModule):
    def __init__(self, in_feat_dim, in_dynamic_rg_dim, in_static_rg_dim, 
                    time_steps, feature_dim, head_num, k, F):
        super(SceneTransformer, self).__init__()
        self.in_feat_dim = in_feat_dim
        self.in_dynamic_rg_dim = in_dynamic_rg_dim
        self.in_static_rg_dim = in_static_rg_dim
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k
        self.F = F

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
        no_nonpad_mask *= ~states_padding_mask_batch[:,10]

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
        loss_ = Loss(gt.unsqueeze(1).repeat(1,6,1), prediction)
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
        no_nonpad_mask *= ~states_padding_mask_batch[:,10]

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
        
        Loss = nn.MSELoss(reduction='none')
        loss_ = Loss(gt.unsqueeze(1).repeat(1,6,1), prediction)
        loss_ = torch.min(torch.mean(torch.mean(loss_, dim=0),dim=-1))

        self.log_dict({'val_loss': loss_})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9,0.999))
        return optimizer
