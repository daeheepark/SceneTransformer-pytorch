from subprocess import call
import os.path
import glob
from tqdm import tqdm

tfrecord_path = 'data/uncompressed/tf_example/training'
idx_path = 'data/tf_exmple_idxs/training/bs4'
batch_size = 4

# tfrecord_path = '/home/user/daehee/SceneTransformer-pytorch/data/single_sample/data'
# idx_path = '/home/user/daehee/SceneTransformer-pytorch/data/single_sample/idxs'
# batch_size = 16

for tfrecord in tqdm(glob.glob(tfrecord_path+'/*')):
    idxname = idx_path + '/' + tfrecord.split('/')[-1]
    #if not os.path.isfile(idxname):  
    call(["tfrecord2idx", tfrecord, idxname])
