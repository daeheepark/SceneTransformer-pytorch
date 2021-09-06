from subprocess import call
import os.path
import glob
from tqdm import tqdm

tfrecord_path = './data/tfrecords_tmp'
idx_path = './data/idxs_tmp'
batch_size=2

for tfrecord in tqdm(glob.glob(tfrecord_path+'/*')):
    idxname = idx_path + '/' + tfrecord.split('/')[-1]
    #if not os.path.isfile(idxname):  
    call(["tfrecord2idx", tfrecord, idxname])
