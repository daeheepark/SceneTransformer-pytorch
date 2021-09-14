import math
import os, glob
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import HTML
import itertools
import tensorflow as tf

import torch
from torch.utils.data import Dataset, IterableDataset

from google.protobuf import text_format
# from waymo_open_dataset.metrics.ops import py_metrics_ops
# from waymo_open_dataset.metrics.python import config_util_py as config_util
# from waymo_open_dataset.protos import motion_metrics_pb2

# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/future/state':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/future/valid':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/future/x':
        tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
    'traffic_light_state/future/y':
        tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
    'traffic_light_state/future/z':
        tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)

def _parse(value):
  decoded_example = tf.io.parse_single_example(value, features_description)

  past_states = tf.stack([
      decoded_example['state/past/x'], decoded_example['state/past/y'],
      decoded_example['state/past/length'], decoded_example['state/past/width'],
      decoded_example['state/past/bbox_yaw'],
      decoded_example['state/past/velocity_x'],
      decoded_example['state/past/velocity_y']
  ], -1)

  cur_states = tf.stack([
      decoded_example['state/current/x'], decoded_example['state/current/y'],
      decoded_example['state/current/length'],
      decoded_example['state/current/width'],
      decoded_example['state/current/bbox_yaw'],
      decoded_example['state/current/velocity_x'],
      decoded_example['state/current/velocity_y']
  ], -1)

  input_states = tf.concat([past_states, cur_states], 1)[..., :2]

  future_states = tf.stack([
      decoded_example['state/future/x'], decoded_example['state/future/y'],
      decoded_example['state/future/length'],
      decoded_example['state/future/width'],
      decoded_example['state/future/bbox_yaw'],
      decoded_example['state/future/velocity_x'],
      decoded_example['state/future/velocity_y']
  ], -1)

  gt_future_states = tf.concat([past_states, cur_states, future_states], 1)

  past_is_valid = decoded_example['state/past/valid'] > 0
  current_is_valid = decoded_example['state/current/valid'] > 0
  future_is_valid = decoded_example['state/future/valid'] > 0
  gt_future_is_valid = tf.concat(
      [past_is_valid, current_is_valid, future_is_valid], 1)

  # If a sample was not seen at all in the past, we declare the sample as
  # invalid.
  sample_is_valid = tf.reduce_any(
      tf.concat([past_is_valid, current_is_valid], 1), 1)

  inputs = {
      'input_states': input_states,
      'gt_future_states': gt_future_states,
      'gt_future_is_valid': gt_future_is_valid,
      'object_type': decoded_example['state/type'],
      'tracks_to_predict': decoded_example['state/tracks_to_predict'] > 0,
      'sample_is_valid': sample_is_valid,
  }
  return inputs

class WaymoTFDataset(IterableDataset):
    def __init__(self, filenames, shuffle=False):
        self.tfdataset = tf.data.TFRecordDataset(filenames)
        if shuffle:
            self.tfdataset = self.tfdataset.shuffle(1)
        self.num_samples = sum(1 for _ in self.tfdataset)

    # def __len__(self):
    #     return self.num_samples

    def __iter__(self):
        return self.tfdataset.as_numpy_iterator()

def waymo_collate_fn(batch, time_steps=10, current_step=3, sampling_time=0.5, GD=16, GS=1400, is_local=True, halfwidth=100, only_veh=True): # GS = max number of static roadgraph element (1400), GD = max number of dynamic roadgraph (16)
    
    sampling_freq = int(sampling_time/0.1)
    assert sampling_freq == sampling_time/0.1
    states_batch = np.array([]).reshape(-1,time_steps,9)

    states_padding_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_BP_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_CBP_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_GDP_batch =np.array([]).reshape(-1,time_steps)

    roadgraph_feat_batch = np.array([]).reshape(-1,time_steps,6)
    roadgraph_padding_batch = np.array([]).reshape(-1,time_steps)

    traffic_light_feat_batch = np.array([]).reshape(-1,time_steps,3)
    traffic_light_padding_batch = np.array([]).reshape(-1,time_steps)

    num_agents = np.array([])
    num_rg = np.array([])
    num_tl = np.array([])

    for data in batch:
        data = tf.io.parse_single_example(data, features_description)
        # State of Agents
        past_states = np.stack((data['state/past/x'],data['state/past/y'],data['state/past/bbox_yaw'],
                                    data['state/past/velocity_x'],data['state/past/velocity_y'],data['state/past/vel_yaw'],
                                        data['state/past/width'],data['state/past/length'],data['state/past/timestamp_micros']), axis=-1)
        past_states_valid = data['state/past/valid'] > 0
        current_states = np.stack((data['state/current/x'],data['state/current/y'],data['state/current/bbox_yaw'],
                                    data['state/current/velocity_x'],data['state/current/velocity_y'],data['state/current/vel_yaw'],
                                        data['state/current/width'],data['state/current/length'],data['state/current/timestamp_micros']), axis=-1)
        current_states_valid = data['state/current/valid'] > 0
        future_states = np.stack((data['state/future/x'],data['state/future/y'],data['state/future/bbox_yaw'],
                                    data['state/future/velocity_x'],data['state/future/velocity_y'],data['state/future/vel_yaw'],
                                        data['state/future/width'],data['state/future/length'],data['state/future/timestamp_micros']), axis=-1)
        future_states_valid = data['state/future/valid'] > 0

        states_feat = np.concatenate((past_states,current_states,future_states),axis=1)                             # [A,T,D]
        states_padding = ~np.concatenate((past_states_valid,current_states_valid,future_states_valid), axis=1) # [A,T]
        states_feat = states_feat[:,::sampling_freq,:][:,:time_steps,:]
        states_padding = states_padding[:,::sampling_freq][:,:time_steps]
        states_feat[:,:,-1] = np.where(states_padding, states_feat[:,:,-1], states_feat[:,:,-1]/1e6)

        # basic_mask = np.zeros((len(states_feat),time_steps)).astype(np.bool_)
        states_hidden_BP = np.ones((len(states_feat),time_steps)).astype(np.bool_)
        states_hidden_BP[:,:current_step+1] = False
        sdvidx = np.where(data['state/is_sdc'] == 1)[0][0]
        states_hidden_CBP = np.ones((len(states_feat),time_steps)).astype(np.bool_)
        states_hidden_CBP[:,:current_step+1] = False
        states_hidden_CBP[sdvidx,:] = False
        states_hidden_GDP = np.ones((len(states_feat),time_steps)).astype(np.bool_)
        states_hidden_GDP[:,:current_step+1] = False
        states_hidden_GDP[sdvidx,-1] = False
        # states_hidden_mask_CDP = np.zeros((len(states_feat),time_steps)).astype(np.bool_)
        
        # Static Road Graph
        roadgraph_feat = np.concatenate((data['roadgraph_samples/xyz'][:,:2], data['roadgraph_samples/dir'][:,:2],
                                            data['roadgraph_samples/type'], data['roadgraph_samples/id']), axis=-1)
        roadgraph_valid = data['roadgraph_samples/valid'] > 0

        roadgraph_feat = roadgraph_feat[roadgraph_valid[:,0]]
        roadgraph_valid = np.ones((roadgraph_feat.shape[0],1)).astype(np.bool_)

        roadgraph_feat = np.tile(roadgraph_feat[:,None,:],(1,time_steps,1))
        roadgraph_valid = np.tile(roadgraph_valid,(1,time_steps))
        roadgraph_padding = ~roadgraph_valid

        # Dynamic Road Graph
        traffic_light_states_past = np.stack((np.transpose(data['traffic_light_state/past/x']),np.transpose(data['traffic_light_state/past/y']),np.transpose(data['traffic_light_state/past/state'])),axis=-1)
        traffic_light_valid_past = np.transpose(data['traffic_light_state/past/valid']) > 0
        traffic_light_states_current = np.stack((np.transpose(data['traffic_light_state/current/x']),np.transpose(data['traffic_light_state/current/y']),np.transpose(data['traffic_light_state/current/state'])),axis=-1)
        traffic_light_valid_current = np.transpose(data['traffic_light_state/current/valid']) > 0
        traffic_light_states_future = np.stack((np.transpose(data['traffic_light_state/future/x']),np.transpose(data['traffic_light_state/future/y']),np.transpose(data['traffic_light_state/future/state'])),axis=-1)
        traffic_light_valid_future = np.transpose(data['traffic_light_state/future/valid']) > 0

        traffic_light_feat = np.concatenate((traffic_light_states_past,traffic_light_states_current,traffic_light_states_future),axis=1)
        traffic_light_valid = np.concatenate((traffic_light_valid_past,traffic_light_valid_current,traffic_light_valid_future),axis=1)
        traffic_light_feat = traffic_light_feat[:,::sampling_freq,:][:,:time_steps,:]
        traffic_light_valid = traffic_light_valid[:,::sampling_freq][:,:time_steps]
        traffic_light_padding = ~traffic_light_valid

        if is_local:
            center_x, center_y = states_feat[sdvidx,current_step,:2]
            
            states_feat[:,:,:2] = np.where(np.tile(states_padding[:,:,None], (1,1,2)), states_feat[:,:,:2], (states_feat[:,:,:2]-np.array([center_x,center_y]))/halfwidth)
            roadgraph_feat[:,:,:2] = np.where(np.tile(roadgraph_padding[:,:,None], (1,1,2)), roadgraph_feat[:,:,:2], (roadgraph_feat[:,:,:2]-np.array([center_x,center_y]))/halfwidth)
            traffic_light_feat[:,:,:2] = np.where(np.tile(traffic_light_padding[:,:,None], (1,1,2)), traffic_light_feat[:,:,:2], (traffic_light_feat[:,:,:2]-np.array([center_x,center_y]))/halfwidth)
            
            agent_xy_mask = (states_feat[:,:,:2] >= -1) * (states_feat[:,:,:2] <= 1)
            agent_xy_mask = agent_xy_mask[:,:,0]*agent_xy_mask[:,:,1]
            states_feat[~agent_xy_mask] = -1
            states_padding += ~agent_xy_mask            
            
            rg_xy_mask = (roadgraph_feat[:,:,:2] >= -1) * (roadgraph_feat[:,:,:2] <= 1)
            rg_xy_mask = rg_xy_mask[:,:,0]*rg_xy_mask[:,:,1]
            roadgraph_feat = roadgraph_feat[rg_xy_mask[:,0]]
            roadgraph_padding = np.zeros((roadgraph_feat.shape[0],time_steps)).astype(np.bool_)

            tl_xy_mask = (traffic_light_feat[:,:,:2] >= -1) * (traffic_light_feat[:,:,:2] <= 1)
            tl_xy_mask = tl_xy_mask[:,:,0]*tl_xy_mask[:,:,1]
            traffic_light_feat[~tl_xy_mask] = -1
            traffic_light_padding += ~tl_xy_mask

        if only_veh:
            # Mask only vehicles
            agent_types = data['state/type'] 
            agent_type_mask = agent_types==1
            states_feat = states_feat[agent_type_mask]
            states_padding = states_padding[agent_type_mask]

            states_hidden_BP = states_hidden_BP[agent_type_mask]
            states_hidden_CBP = states_hidden_CBP[agent_type_mask]
            states_hidden_GDP = states_hidden_GDP[agent_type_mask]


        if roadgraph_feat.shape[0] > GS:
            spacing = roadgraph_feat.shape[0] // GS
            roadgraph_feat = roadgraph_feat[::spacing, :]
            remove_num = len(roadgraph_feat) - GS
            roadgraph_mask2 = np.full(len(roadgraph_feat), True)
            idx_remove = np.random.choice(range(len(roadgraph_feat)), remove_num, replace=False)
            roadgraph_mask2[idx_remove] = False
            roadgraph_feat = roadgraph_feat[roadgraph_mask2]
            roadgraph_padding = np.zeros((GS,time_steps)).astype(np.bool_)

        states_any_mask = states_padding.sum(axis=-1) != time_steps
        states_feat = states_feat[states_any_mask]
        states_padding = states_padding[states_any_mask]
        roadgarph_any_mask = roadgraph_padding.sum(axis=-1) != time_steps
        roadgraph_feat = roadgraph_feat[roadgarph_any_mask]
        roadgraph_padding = roadgraph_padding[roadgarph_any_mask]
        tl_any_mask = traffic_light_padding.sum(axis=-1) != time_steps
        traffic_light_feat = traffic_light_feat[tl_any_mask]
        traffic_light_padding = traffic_light_padding[tl_any_mask]

        states_hidden_BP = states_hidden_BP[states_any_mask]
        states_hidden_CBP = states_hidden_CBP[states_any_mask]
        states_hidden_GDP = states_hidden_GDP[states_any_mask]

        num_agents = np.append(num_agents, len(states_feat))
        num_rg = np.append(num_rg, roadgraph_feat.shape[0])
        num_tl = np.append(num_tl, traffic_light_feat.shape[0])

        # Concat across batch
        states_batch = np.concatenate((states_batch,states_feat), axis=0)
        states_padding_batch = np.concatenate((states_padding_batch,states_padding), axis=0)

        states_hidden_BP_batch = np.concatenate((states_hidden_BP_batch,states_hidden_BP), axis=0)
        states_hidden_CBP_batch = np.concatenate((states_hidden_CBP_batch,states_hidden_CBP), axis=0)
        states_hidden_GDP_batch =np.concatenate((states_hidden_GDP_batch,states_hidden_GDP), axis=0)

        roadgraph_feat_batch = np.concatenate((roadgraph_feat_batch, roadgraph_feat), axis=0)
        roadgraph_padding_batch = np.concatenate((roadgraph_padding_batch, roadgraph_padding), axis=0)

        traffic_light_feat_batch = np.concatenate((traffic_light_feat_batch, traffic_light_feat), axis=0)
        traffic_light_padding_batch = np.concatenate((traffic_light_padding_batch, traffic_light_padding), axis=0)

    num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int32)
    num_rg_accum = np.cumsum(np.insert(num_rg,0,0)).astype(np.int32)
    num_tl_accum = np.cumsum(np.insert(num_tl,0,0)).astype(np.int32)

    agents_batch_mask = np.ones((num_agents_accum[-1],num_agents_accum[-1])) # padding. 1 -> padded (ignore att) / 0 -> non-padded (do att)
    agent_rg_mask = np.ones((num_agents_accum[-1],num_rg_accum[-1]))
    agent_traffic_mask = np.ones((num_agents_accum[-1],num_tl_accum[-1]))

    for i in range(len(num_agents)):
        agents_batch_mask[num_agents_accum[i]:num_agents_accum[i+1], num_agents_accum[i]:num_agents_accum[i+1]] = 0
        agent_rg_mask[num_agents_accum[i]:num_agents_accum[i+1], num_rg_accum[i]:num_rg_accum[i+1]] = 0
        agent_traffic_mask[num_agents_accum[i]:num_agents_accum[i+1], num_tl_accum[i]:num_tl_accum[i+1]] = 0

    states_batch = torch.FloatTensor(states_batch)
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)
    states_padding_batch = torch.BoolTensor(states_padding_batch)
    states_hidden_BP_batch = torch.BoolTensor(states_hidden_BP_batch)
    states_hidden_CBP_batch = torch.BoolTensor(states_hidden_CBP_batch)
    states_hidden_GDP_batch = torch.BoolTensor(states_hidden_GDP_batch)
    
    roadgraph_feat_batch = torch.FloatTensor(roadgraph_feat_batch)
    roadgraph_padding_batch = torch.BoolTensor(roadgraph_padding_batch)
    traffic_light_feat_batch = torch.FloatTensor(traffic_light_feat_batch)
    traffic_light_padding_batch = torch.BoolTensor(traffic_light_padding_batch)

    agent_rg_mask = torch.BoolTensor(agent_rg_mask)
    agent_traffic_mask = torch.BoolTensor(agent_traffic_mask)   
        
    return (states_batch, agents_batch_mask, states_padding_batch, 
                (states_hidden_BP_batch, states_hidden_CBP_batch, states_hidden_GDP_batch), 
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                        agent_rg_mask, agent_traffic_mask)