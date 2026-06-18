import argparse
import os
import numpy as np
from numpy.lib.format import open_memmap
import faiss
from faiss.contrib.ondisk import merge_ondisk
import time
import ctypes
import json
from tqdm import trange, tqdm

TRAINING_NUM = 1000000
MAX_BLOCK_SIZE = 250000000

parser = argparse.ArgumentParser()
parser.add_argument('--dstore-path', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--index-file', type=str, help='file to write the faiss index')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore-fp16', default=False, action='store_true')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
parser.add_argument('--code-size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
parser.add_argument('--num-keys-to-add-at-a-time', default=1000000, type=int,
                    help='can only load a certain amount of data to dstore at a time.')
parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--output-dir', type=str, default=None)
parser.add_argument('--cpu-train', default=False, action='store_true',
                    help='train the IVF-PQ index on CPU, then write the trained CPU index')

args = parser.parse_args()
print(args)
# continual = args.continual
# if args.date.isdigit():
#     pre = int(args.date) - 1
fp = 16 if args.dstore_fp16 else 32

*_, dataset, _, dstore_prefix = args.dstore_path.split('/')
info_list = dstore_prefix.split('_')
ckpt_name, date, prune_param = info_list[0], info_list[1], info_list[-1]

a = json.load(open(f'{args.output_dir}/size.json', 'r'))
dstore_size = int(a[dataset][ckpt_name][date][prune_param])
print(dstore_size)

keys = open_memmap(args.dstore_path+'_keys.npy', mode='r+')[:dstore_size]
vals = open_memmap(args.dstore_path+'_vals.npy', mode='r+')
# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
# madvise = ctypes.CDLL("libc.so.6").madvise # Only Linux
# madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
# madvise.restype = ctypes.c_int
# assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

# Train index
if not os.path.exists(args.index_file+".trained"):
    # nThreads = 1
    # faiss.omp_set_num_threads(nThreads)
    # Initialize faiss index
    # ngpus = faiss.get_num_gpus()
    # print("number of GPUs:", ngpus)
    quantizer = faiss.IndexFlatL2(args.dimension)
    cpu_index = faiss.IndexIVFPQ(quantizer, args.dimension,
        args.ncentroids, args.code_size, 8)
    cpu_train = args.cpu_train or os.environ.get('KNNLM_FAISS_TRAIN_ON_CPU') == '1'
    if cpu_train:
        print('Using cpu faiss index')
        index = cpu_index
    else:
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        res = faiss.StandardGpuResources()
        print('Using gpu faiss index')
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
        # multi-GPU will slow down training
        # co = faiss.GpuMultipleClonerOptions()
        # index = faiss.index_cpu_to_all_gpus(cpu_index, co)
    index.nprobe = args.probe

    print('Training Index')
    np.random.seed(args.seed)
    start = time.time()
    random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(TRAINING_NUM, vals.shape[0])], replace=False)
    # Faiss does not handle adding keys in fp16 as of writing this.
    random_total_keys = keys[random_sample].astype(np.float32)
    print('Reading took {} s'.format(time.time() - start))
    
    index.train(random_total_keys)
    print('Trained. Writing index to {}'.format(args.index_file+".trained"))
    if not cpu_train:
        cpu_index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(cpu_index, args.index_file+".trained")
    print('Training index took {} s in all'.format(time.time()-start))
