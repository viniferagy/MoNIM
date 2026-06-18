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
parser.add_argument('--num-shards', type=int, default=None)

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
# print(dstore_size)

# keys = open_memmap(args.dstore_path+'_keys.npy', mode='r+')[:dstore_size]
# vals = open_memmap(args.dstore_path+'_vals.npy', mode='r+')

# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
# madvise = ctypes.CDLL("libc.so.6").madvise # Only Linux
# madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
# madvise.restype = ctypes.c_int
# assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

print('Merging')
index_trained = faiss.read_index(args.index_file+".trained")
for b in range(args.num_shards):
    a = faiss.read_index(args.index_file + f'.{b}')
    index_trained.merge_from(a, add_id=index_trained.ntotal)
print('Writing to ' + args.index_file)
faiss.write_index(index_trained, args.index_file)
print('Now the index has {} vectors in total'.format(index_trained.ntotal))