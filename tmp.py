import tensorflow as tf
from tqdm import tqdm
import glob, os, sys

from torch.utils.data import DataLoader, Dataset, IterableDataset

from datautil.waymo_tfrecord_dataset import WaymoTFDataset, waymo_collate_fn
filenames = glob.glob('data/tfrecords'+'/*')
dataset = WaymoTFDataset(filenames)
loader = DataLoader(dataset, batch_size=1, collate_fn=waymo_collate_fn)

from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn, waymo_worker_fn
filenames = 'data/tf_example/training'
idxs = 'data/idxs_training_bs_8'
bs = 8

# filename = '/home/user/daehee/SceneTransformer-pytorch/datautil/tmp.txt'
# if os.path.isfile(filename):
#     os.remove(filename)
# f = open(filename, 'w')
# for ep in range(2):
#     for it, d in enumerate(tqdm(loader)):
#         f.write(f'{ep} {it} : '+str(d[0][0][0][:2])+'\n')
#         if it % 100 == 0:
#             print(ep, ' ', it, ' : ', d[0][0][0][:2])
#     dataset.tfdataset = dataset.tfdataset.shuffle(1)
# f.close()
# sys.exit()
