import torch
import torch.nn as nn
import torch.nn.functional as FU

from model.utils import *

from collections import OrderedDict

class Decoder(nn.Module):
    def __init__(self, device, time_steps=91, feature_dim=256, head_num=4, k=4, F=6):
        super().__init__()
        self.device = device
        self.time_steps = time_steps                # T
        self.feature_dim = feature_dim              # D
        self.head_num = head_num                    # H
        self.k = k
        self.F = F

        onehots_ = torch.tensor(range(F))
        self.onehots_ = FU.one_hot(onehots_, num_classes=F).to(self.device)

        self.layer_T = nn.Sequential(nn.Linear(self.feature_dim+self.F,feature_dim), nn.ReLU())

        self.layer_U = SelfAttLayer_dec(self.device,self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_V = SelfAttLayer_dec(self.device,self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)

        self.layer_W = SelfAttLayer_dec(self.device,self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_X = SelfAttLayer_dec(self.device,self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)

        self.layer_Y = nn.LayerNorm(self.feature_dim)

        self.layer_Z1 = nn.Sequential(nn.Linear(self.feature_dim,6), nn.ReLU(), Permute4Batchnorm((1,2,0,3)),
                                        nn.BatchNorm2d(self.feature_dim), Permute4Batchnorm((2,0,1,3))) 
        self.layer_Z2 = nn.Sequential(nn.Linear(6,7), nn.ReLU())

    def forward(self, state_feat, batch_mask, padding_mask, hidden_mask):
        A,T,D = state_feat.shape
        assert (T==self.time_steps and D==self.feature_dim)

        onehots_ = self.onehots_.view(self.F,1,1,self.F).repeat(1,A,T,1)
        x = state_feat.unsqueeze(0).repeat(self.F,1,1,1)

        x = torch.cat((x,onehots_),dim=-1)
        x = self.layer_T(x)

        x = self.layer_U(x,batch_mask=batch_mask, padding_mask=padding_mask)
        x = self.layer_V(x,batch_mask=batch_mask, padding_mask=padding_mask)
        
        x = self.layer_W(x,batch_mask=batch_mask, padding_mask=padding_mask)
        x = self.layer_X(x,batch_mask=batch_mask, padding_mask=padding_mask)

        x = self.layer_Y(x)
        x = self.layer_Z1(x)
        x = self.layer_Z2(x)

        return x