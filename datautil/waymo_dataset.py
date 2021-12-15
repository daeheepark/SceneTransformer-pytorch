import math
import os, sys
import uuid
import time
import random

import numpy as np
import torch
try:
    from tfrecordutils.dataset import MultiTFRecordDataset
    from tfrecordutils import reader
    from tfrecordutils import iterator_utils
except:
    from os import path
    print(path.dirname( path.dirname( path.abspath(__file__) ) ))
    sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    from tfrecordutils.dataset import MultiTFRecordDataset
    from tfrecordutils import reader
    from tfrecordutils import iterator_utils


# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir':
        'float',
    'roadgraph_samples/id':
        'int',
    'roadgraph_samples/type':
        'int',
    'roadgraph_samples/valid':
        'int',
    'roadgraph_samples/xyz':
        'float',
}

# Features of other agents.
state_features = {
    'state/id':
        'float',
    'state/type':
        'float',
    'state/is_sdc':
        'int',
    'state/tracks_to_predict':
        'int',
    'state/current/bbox_yaw':
        'float',
    'state/current/height':
        'float',
    'state/current/length':
        'float',
    'state/current/timestamp_micros':
        'int',
    'state/current/valid':
        'int',
    'state/current/vel_yaw':
        'float',
    'state/current/velocity_x':
        'float',
    'state/current/velocity_y':
        'float',
    'state/current/width':
        'float',
    'state/current/x':
        'float',
    'state/current/y':
        'float',
    'state/current/z':
        'float',
    'state/future/bbox_yaw':
        'float',
    'state/future/height':
        'float',
    'state/future/length':
        'float',
    'state/future/timestamp_micros':
        'int',
    'state/future/valid':
        'int',
    'state/future/vel_yaw':
        'float',
    'state/future/velocity_x':
        'float',
    'state/future/velocity_y':
        'float',
    'state/future/width':
        'float',
    'state/future/x':
        'float',
    'state/future/y':
        'float',
    'state/future/z':
        'float',
    'state/past/bbox_yaw':
        'float',
    'state/past/height':
        'float',
    'state/past/length':
        'float',
    'state/past/timestamp_micros':
        'int',
    'state/past/valid':
        'int',
    'state/past/vel_yaw':
        'float',
    'state/past/velocity_x':
        'float',
    'state/past/velocity_y':
        'float',
    'state/past/width':
        'float',
    'state/past/x':
        'float',
    'state/past/y':
        'float',
    'state/past/z':
        'float',
}

traffic_light_features = {
    'traffic_light_state/current/state':
        'int',
    'traffic_light_state/current/valid':
        'int',
    'traffic_light_state/current/x':
        'float',
    'traffic_light_state/current/y':
        'float',
    'traffic_light_state/current/z':
        'float',
    'traffic_light_state/past/state':
        'int',
    'traffic_light_state/past/valid':
        'int',
    'traffic_light_state/past/x':
        'float',
    'traffic_light_state/past/y':
        'float',
    'traffic_light_state/past/z':
        'float',
    'traffic_light_state/future/state':
        'int',
    'traffic_light_state/future/valid':
        'int',
    'traffic_light_state/future/x':
        'float',
    'traffic_light_state/future/y':
        'float',
    'traffic_light_state/future/z':
        'float',
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)


# Example field definition
roadgraph_transforms = {
    'roadgraph_samples/dir':
        lambda x : np.reshape(x,(20000,3)),
    'roadgraph_samples/id':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/type':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/valid':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/xyz':
        lambda x : np.reshape(x,(20000,3)),
}

# Features of other agents.
state_transforms = {
    'state/id':
        lambda x : np.reshape(x,(128,)),
    'state/type':
        lambda x : np.reshape(x,(128,)),
    'state/is_sdc':
        lambda x : np.reshape(x,(128,)),
    'state/tracks_to_predict':
        lambda x : np.reshape(x,(128,)),
    'state/current/bbox_yaw':
        lambda x : np.reshape(x,(128,1)),
    'state/current/height':
        lambda x : np.reshape(x,(128,1)),
    'state/current/length':
        lambda x : np.reshape(x,(128,1)),
    'state/current/timestamp_micros':
        lambda x : np.reshape(x,(128,1)),
    'state/current/valid':
        lambda x : np.reshape(x,(128,1)),
    'state/current/vel_yaw':
        lambda x : np.reshape(x,(128,1)),
    'state/current/velocity_x':
        lambda x : np.reshape(x,(128,1)),
    'state/current/velocity_y':
        lambda x : np.reshape(x,(128,1)),
    'state/current/width':
        lambda x : np.reshape(x,(128,1)),
    'state/current/x':
        lambda x : np.reshape(x,(128,1)),
    'state/current/y':
        lambda x : np.reshape(x,(128,1)),
    'state/current/z':
        lambda x : np.reshape(x,(128,1)),
    'state/future/bbox_yaw':
        lambda x : np.reshape(x,(128,80)),
    'state/future/height':
        lambda x : np.reshape(x,(128,80)),
    'state/future/length':
        lambda x : np.reshape(x,(128,80)),
    'state/future/timestamp_micros':
        lambda x : np.reshape(x,(128,80)),
    'state/future/valid':
        lambda x : np.reshape(x,(128,80)),
    'state/future/vel_yaw':
        lambda x : np.reshape(x,(128,80)),
    'state/future/velocity_x':
        lambda x : np.reshape(x,(128,80)),
    'state/future/velocity_y':
        lambda x : np.reshape(x,(128,80)),
    'state/future/width':
        lambda x : np.reshape(x,(128,80)),
    'state/future/x':
        lambda x : np.reshape(x,(128,80)),
    'state/future/y':
        lambda x : np.reshape(x,(128,80)),
    'state/future/z':
        lambda x : np.reshape(x,(128,80)),
    'state/past/bbox_yaw':
        lambda x : np.reshape(x,(128,10)),
    'state/past/height':
        lambda x : np.reshape(x,(128,10)),
    'state/past/length':
        lambda x : np.reshape(x,(128,10)),
    'state/past/timestamp_micros':
        lambda x : np.reshape(x,(128,10)),
    'state/past/valid':
        lambda x : np.reshape(x,(128,10)),
    'state/past/vel_yaw':
        lambda x : np.reshape(x,(128,10)),
    'state/past/velocity_x':
        lambda x : np.reshape(x,(128,10)),
    'state/past/velocity_y':
        lambda x : np.reshape(x,(128,10)),
    'state/past/width':
        lambda x : np.reshape(x,(128,10)),
    'state/past/x':
        lambda x : np.reshape(x,(128,10)),
    'state/past/y':
        lambda x : np.reshape(x,(128,10)),
    'state/past/z':
        lambda x : np.reshape(x,(128,10)),
}

traffic_light_transforms = {
    'traffic_light_state/current/state':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/valid':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/x':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/y':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/z':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/past/state':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/valid':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/x':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/y':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/z':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/future/state':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/valid':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/x':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/y':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/z':
        lambda x : np.reshape(x,(80,16)),
}

features_transforms = {}
features_transforms.update(roadgraph_transforms)
features_transforms.update(state_transforms)
features_transforms.update(traffic_light_transforms)

def transform_func(feature):
    transform = features_transforms
    keys = transform.keys()
    for key in keys:
        func = transform[key]
        feat = feature[key]
        feature[key] = func(feat)
    return feature

class WaymoDataset(MultiTFRecordDataset):
    def __init__(self, tfrecord_dir, idx_dir, shuffle_queue_size):
        super(WaymoDataset, self).__init__(data_pattern=tfrecord_dir+'/{}',index_pattern=idx_dir+'/{}',splits={}, 
                                            shuffle_queue_size=shuffle_queue_size, infinite=False)
        self.splits = {}
        fnlist = os.listdir(self.data_pattern.split('{}')[0])
        for fn in fnlist:
            self.splits[fn] = 1/len(fnlist)

        # if shuffle_first:
        #     splitsl = list(self.splits.items())
        #     random.shuffle(splitsl)
        #     self.splits = dict(splitsl)
        
        self.description = features_description
        self.sequence_discription = None
        self.transform = transform_func

        if self.index_pattern is not None:
            self.num_samples = sum(
                sum(1 for _ in open(self.index_pattern.format(split)))
                for split in self.splits
            )
        else:
            self.num_samples = None

    def __len__(self):
        if self.num_samples is not None:
            return int(self.num_samples)
        else:
            raise NotImplementedError()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
            # dataset = worker_info.dataset
            worker_id = worker_info.id
            assert len(self.splits) >= worker_info.num_workers, 'num_workers should be smaller than number of splits'
            split_size = len(self.splits) // worker_info.num_workers
            overall_end = len(self.splits)
            if not len(self.splits) - (worker_id+1)*split_size < split_size:
                self.splits = dict(list(self.splits.items())[worker_id*split_size:(worker_id+1)*split_size])
            else:
                self.splits = dict(list(self.splits.items())[worker_id*split_size:overall_end])
        it = reader.multi_tfrecord_loader(data_pattern=self.data_pattern,
                                          index_pattern=self.index_pattern,
                                          splits=self.splits,
                                          description=self.description,
                                          sequence_description=self.sequence_description,
                                          compression_type=self.compression_type,
                                          infinite=self.infinite,
                                         )
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform, it)
        return it


def waymo_collate_fn(batch, time_steps=10, current_step=3, sampling_time=0.5, GD=16, GS=1400, is_local=True, halfwidth=50, only_veh=True, hidden='BP'): # GS = max number of static roadgraph element (1400), GD = max number of dynamic roadgraph (16)
    sampling_freq = int(sampling_time/0.1)
    assert sampling_freq == sampling_time/0.1
    states_batch = np.array([]).reshape(-1,time_steps,9)

    states_padding_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_batch = np.array([]).reshape(-1,time_steps)
    
    roadgraph_feat_batch = np.array([]).reshape(-1,time_steps,6)
    roadgraph_padding_batch = np.array([]).reshape(-1,time_steps)

    traffic_light_feat_batch = np.array([]).reshape(-1,time_steps,3)
    traffic_light_padding_batch = np.array([]).reshape(-1,time_steps)

    num_agents = np.array([])
    num_rg = np.array([])
    num_tl = np.array([])

    sdc_masks = []
    center_ps = []

    for data in batch:
        # State of Agents
        past_states = np.stack((data['state/past/x'],data['state/past/y'],data['state/past/bbox_yaw'],
                                    data['state/past/velocity_x'],data['state/past/velocity_y'],data['state/past/vel_yaw'],
                                        data['state/past/width'],data['state/past/length'],data['state/past/timestamp_micros']), axis=-1)
        past_states_valid = data['state/past/valid'] > 0.
        current_states = np.stack((data['state/current/x'],data['state/current/y'],data['state/current/bbox_yaw'],
                                    data['state/current/velocity_x'],data['state/current/velocity_y'],data['state/current/vel_yaw'],
                                        data['state/current/width'],data['state/current/length'],data['state/current/timestamp_micros']), axis=-1)
        current_states_valid = data['state/current/valid'] > 0.
        future_states = np.stack((data['state/future/x'],data['state/future/y'],data['state/future/bbox_yaw'],
                                    data['state/future/velocity_x'],data['state/future/velocity_y'],data['state/future/vel_yaw'],
                                        data['state/future/width'],data['state/future/length'],data['state/future/timestamp_micros']), axis=-1)
        future_states_valid = data['state/future/valid'] > 0.

        states_feat = np.concatenate((past_states,current_states,future_states),axis=1)                             # [A,T,D]
        states_padding = ~np.concatenate((past_states_valid,current_states_valid,future_states_valid), axis=1)      # [A,T]
        states_feat = states_feat[:,::sampling_freq,:][:,:time_steps,:]
        states_padding = states_padding[:,::sampling_freq][:,:time_steps]
        states_feat[:,:,-1] = np.where(states_padding, states_feat[:,:,-1], states_feat[:,:,-1]/1e6)

        sdc_mask = data['state/is_sdc'] == 1
        sdvidx = np.where(sdc_mask)[0][0]

        # basic_mask = np.zeros((len(states_feat),time_steps)).astype(np.bool_)
        if hidden == 'BP':
            states_hidden = np.ones((len(states_feat),time_steps)).astype(np.bool_)
            states_hidden[:,:current_step+1] = False
        elif hidden == 'CBP':
            states_hidden = np.ones((len(states_feat),time_steps)).astype(np.bool_)
            states_hidden[:,:current_step+1] = False
            states_hidden[sdvidx,:] = False
        elif hidden == 'GDP':
            states_hidden = np.ones((len(states_feat),time_steps)).astype(np.bool_)
            states_hidden[:,:current_step+1] = False
            states_hidden[sdvidx,-1] = False
        else:
            raise KeyError('hidden should be BP or CBP or GDP')
        
        
        # Static Road Graph
        roadgraph_feat = np.concatenate((data['roadgraph_samples/xyz'][:,:2], data['roadgraph_samples/dir'][:,:2],
                                            data['roadgraph_samples/type'], data['roadgraph_samples/id']), axis=-1)
        roadgraph_valid = data['roadgraph_samples/valid'] > 0.

        roadgraph_feat = roadgraph_feat[roadgraph_valid[:,0]]
        roadgraph_valid = np.ones((roadgraph_feat.shape[0],1)).astype(np.bool_)

        roadgraph_feat = np.tile(roadgraph_feat[:,None,:],(1,time_steps,1))
        roadgraph_valid = np.tile(roadgraph_valid,(1,time_steps))
        roadgraph_padding = ~roadgraph_valid

        # Dynamic Road Graph
        traffic_light_states_past = np.stack((data['traffic_light_state/past/x'].T,data['traffic_light_state/past/y'].T,data['traffic_light_state/past/state'].T),axis=-1)
        traffic_light_valid_past = data['traffic_light_state/past/valid'].T > 0.
        traffic_light_states_current = np.stack((data['traffic_light_state/current/x'].T,data['traffic_light_state/current/y'].T,data['traffic_light_state/current/state'].T),axis=-1)
        traffic_light_valid_current = data['traffic_light_state/current/valid'].T > 0.
        traffic_light_states_future = np.stack((data['traffic_light_state/future/x'].T,data['traffic_light_state/future/y'].T,data['traffic_light_state/future/state'].T),axis=-1)
        traffic_light_valid_future = data['traffic_light_state/future/valid'].T > 0.

        traffic_light_feat = np.concatenate((traffic_light_states_past,traffic_light_states_current,traffic_light_states_future),axis=1)
        traffic_light_valid = np.concatenate((traffic_light_valid_past,traffic_light_valid_current,traffic_light_valid_future),axis=1)
        traffic_light_feat = traffic_light_feat[:,::sampling_freq,:][:,:time_steps,:]
        traffic_light_valid = traffic_light_valid[:,::sampling_freq][:,:time_steps]
        traffic_light_padding = ~traffic_light_valid

        center_p = states_feat[sdvidx,current_step,:2].copy()

        if is_local:
            states_feat[:,:,:2] = np.where(np.tile(states_padding[:,:,None], (1,1,2)), states_feat[:,:,:2], (states_feat[:,:,:2]-center_p)/halfwidth)
            roadgraph_feat[:,:,:2] = np.where(np.tile(roadgraph_padding[:,:,None], (1,1,2)), roadgraph_feat[:,:,:2], (roadgraph_feat[:,:,:2]-center_p)/halfwidth)
            traffic_light_feat[:,:,:2] = np.where(np.tile(traffic_light_padding[:,:,None], (1,1,2)), traffic_light_feat[:,:,:2], (traffic_light_feat[:,:,:2]-center_p)/halfwidth)
            
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

            center_p = states_feat[sdvidx,current_step,:2]

        if only_veh:
            # Mask only vehicles
            agent_types = data['state/type'] 
            agent_type_mask = agent_types==1
            states_feat = states_feat[agent_type_mask]
            states_padding = states_padding[agent_type_mask]
            states_hidden = states_hidden[agent_type_mask]

        if roadgraph_feat.shape[0] > GS:
            spacing = roadgraph_feat.shape[0] // GS
            roadgraph_feat = roadgraph_feat[::spacing, :]
            remove_num = len(roadgraph_feat) - GS
            roadgraph_mask2 = np.full(len(roadgraph_feat), True)
            idx_remove = np.random.choice(range(len(roadgraph_feat)), remove_num, replace=False)
            roadgraph_mask2[idx_remove] = False
            roadgraph_feat = roadgraph_feat[roadgraph_mask2]
            roadgraph_padding = np.zeros((GS,time_steps)).astype(np.bool_)

        # states_any_mask = states_padding.sum(axis=-1) != time_steps
        states_any_mask = (states_padding + states_hidden).sum(-1) != time_steps
        states_feat = states_feat[states_any_mask]
        states_padding = states_padding[states_any_mask]
        states_hidden = states_hidden[states_any_mask]

        roadgarph_any_mask = roadgraph_padding.sum(axis=-1) != time_steps
        roadgraph_feat = roadgraph_feat[roadgarph_any_mask]
        roadgraph_padding = roadgraph_padding[roadgarph_any_mask]

        tl_any_mask = traffic_light_padding.sum(axis=-1) != time_steps
        traffic_light_feat = traffic_light_feat[tl_any_mask]
        traffic_light_padding = traffic_light_padding[tl_any_mask]

        num_agents = np.append(num_agents, len(states_feat))
        num_rg = np.append(num_rg, roadgraph_feat.shape[0])
        num_tl = np.append(num_tl, traffic_light_feat.shape[0])

        sdc_masks.append(torch.BoolTensor(sdc_mask))
        center_ps.append(torch.FloatTensor(center_p))

        # Concat across batch
        states_batch = np.concatenate((states_batch,states_feat), axis=0)
        states_padding_batch = np.concatenate((states_padding_batch,states_padding), axis=0)

        states_hidden_batch = np.concatenate((states_hidden_batch,states_hidden), axis=0)

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

    states_batch = torch.FloatTensor(states_batch)                                  # [A,T,9] -> (x, y, bbox_yaw, vel_x, vel_y, vel_yaw, width, len, time)
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)                         # [A,A]
    states_padding_batch = torch.BoolTensor(states_padding_batch)                   # [A,T]
    states_hidden_batch = torch.BoolTensor(states_hidden_batch)
    # states_hidden_BP_batch = torch.BoolTensor(states_hidden_BP_batch)               # [A,T]
    # states_hidden_CBP_batch = torch.BoolTensor(states_hidden_CBP_batch)             # [A,T]
    # states_hidden_GDP_batch = torch.BoolTensor(states_hidden_GDP_batch)             # [A,T]
    
    roadgraph_feat_batch = torch.FloatTensor(roadgraph_feat_batch)                  # [GS,T,6] -> (x, y, dx, dy, type, id)
    roadgraph_padding_batch = torch.BoolTensor(roadgraph_padding_batch)             # [GS,T]
    traffic_light_feat_batch = torch.FloatTensor(traffic_light_feat_batch)          # [GD,T,3] -> (x, y, state)
    traffic_light_padding_batch = torch.BoolTensor(traffic_light_padding_batch)     # [GD,T]

    agent_rg_mask = torch.BoolTensor(agent_rg_mask)                                 # [A,GS]
    agent_traffic_mask = torch.BoolTensor(agent_traffic_mask)                       # [A,GD]
        
    return {'agt_stat': states_batch, 'agt_batch_msk': agents_batch_mask, 'agt_pad_msk': states_padding_batch, 
            'stat_hid_msk': states_hidden_batch, 
            'rg_feat': roadgraph_feat_batch, 'rg_pad_msk': roadgraph_padding_batch, 
            'tr_feat': traffic_light_feat_batch, 'tr_pad_msk': traffic_light_padding_batch,
            'agt_rg_msk': agent_rg_mask, 'agt_tr_msk': agent_traffic_mask, 'num_accum': (num_agents_accum, num_rg_accum, num_tl_accum),
            'sdc_msk': sdc_masks, 'ctr_p': center_ps}

def waymo_worker_fn(wid):
    worker_info = torch.utils.data.get_worker_info()
    
    dataset = worker_info.dataset
    worker_id = worker_info.id
    assert len(dataset.splits) >= worker_info.num_workers, f'num_workers ({worker_info.num_workers}) should be smaller than number of splits ({len(dataset.splits)})'
    split_size = len(dataset.splits) // worker_info.num_workers

    overall_end = len(dataset.splits)
    if not len(dataset.splits) - (worker_id+1)*split_size < split_size:
        dataset.splits = dict(list(dataset.splits.items())[worker_id*split_size:(worker_id+1)*split_size])
    else:
        dataset.splits = dict(list(dataset.splits.items())[worker_id*split_size:overall_end])

def xy_to_pixel(xy, width, mask=True):
    # assert type(xy) == torch.Tensor

    xy[...,1] *= -1
    xy += int(width/2)
    # mask_x = (xy[...,0] <= width)*(xy[...,0]>=0)
    # mask_y = (xy[...,1] <= width)*(xy[...,1]>=0)

    # xy = xy[mask_x*mask_y]
    # xy = torch.clamp(xy, min=0, max=width)

    return xy

if __name__ == '__main__':
    import hydra, os
    import os.path as osp
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import pytorch_lightning as pl
    @hydra.main(config_path='../conf', config_name='config.yaml')
    def main(cfg):
        pl.seed_everything(cfg.seed)
        filename = '/home/user/daehee/SceneTransformer-pytorch/datautil/tmp2.txt'
        if os.path.isfile(filename):
            os.remove(filename)
        f = open(filename, 'w')
        pwd = hydra.utils.get_original_cwd()
        dataset_train = WaymoDataset(osp.join(pwd, cfg.dataset.train.tfrecords), osp.join(pwd, cfg.dataset.train.idxs), shuffle_queue_size=cfg.dataset.train.batchsize)
        # print(len(dataset_train))
        dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.train.batchsize, collate_fn=waymo_collate_fn, worker_init_fn=waymo_worker_fn, num_workers=3, shuffle=False)
        for ep in range(2):
            for it, d in enumerate(tqdm(dloader_train)):
                # f.write(f'{ep} {it} : '+str(d[0][0][0][:2])+'\n')
                # d_ = torch.reshape(d[4],(-1,1400,10,6))
                sample_rg = d['rg_feat'][0,0,:2]
                # break
                f.write(str(ep) + ' ' + str(it)+' : '+str(sample_rg)+'\n')
                if it % 500 == 0:
                    print(ep, ' ', it, ' : ', d['agt_stat'][0][0])
                # if it == 10:
                #     break
        f.close()
    
    sys.exit(main())

