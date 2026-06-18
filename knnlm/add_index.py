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
import math

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
parser.add_argument('--num-keys-to-add-at-a-time', default=500000, type=int,
                    help='can only load a certain amount of data to dstore at a time.')
parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--output-dir', type=str, default=None)
parser.add_argument('--num-shards', type=int, default=1)
parser.add_argument('--shard-id', type=int, default=0)

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
dstore_size = int(a[dataset][ckpt_name][date][prune_param]) / args.num_shards
dstore_begin = math.ceil(dstore_size*args.shard_id)
dstore_end = math.ceil(dstore_size*(args.shard_id+1))
dstore_size = dstore_end - dstore_begin

keys = open_memmap(args.dstore_path+'_keys.npy', mode='r+')[dstore_begin:dstore_end]
# vals = open_memmap(args.dstore_path+'_vals.npy', mode='r+')

print('Adding Keys')
print('Reading index from {}'.format(args.index_file+".trained"))
index = faiss.read_index(args.index_file+".trained")
co = faiss.GpuClonerOptions()
# co = faiss.GpuMultipleClonerOptions()
co.useFloat16 = True
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index, co)

# multi-GPU will slow down training
# index = faiss.index_cpu_to_all_gpus(index, co)

start = 0
end = -1
block_num = 0

start_time = time.time()
while start < dstore_size:
    # A forced special branch, lest cuda out of dstore
    if end % MAX_BLOCK_SIZE == 0:
        # save current block
        print('We have toooo many keys adding once for all. Begin next block')
        cpu_index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(cpu_index, args.index_file + f'.block{block_num}')
        block_num += 1
        # begin next block
        index = faiss.read_index(args.index_file+".trained")
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        print('-- Next block --')
    
    end = min(dstore_size, start+args.num_keys_to_add_at_a_time)
    to_add = keys[start:end].copy()
    index.add(to_add.astype(np.float32))
    # index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
    start = end

    if end % 1000000 == 0:
        print('Added {} tokens so far, took {} s'.format(end, time.time() - start_time))

print(f"Adding total {end} keys, took {time.time() - start_time} s")
last_index_name = args.index_file + f'.block{block_num}' if block_num > 0 else args.index_file + f'.{args.shard_id}'
print('Writing Index to {}'.format(last_index_name))
start_time = time.time()
cpu_index = faiss.index_gpu_to_cpu(index)
faiss.write_index(cpu_index, last_index_name)
print('Writing index took {} s'.format(time.time()-start_time))

if block_num > 0:
    print('Merging the blocks')
    index_trained = faiss.read_index(args.index_file+".trained")
    index_list = [args.index_file + f'.block{b}' for b in range(block_num+1)]
    merge_ondisk(index_trained, index_list, args.index_file + '.merged')
    print('Writing to ' + args.index_file)
    faiss.write_index(index_trained, args.index_file)
    print('Now the index has {} vectors in total'.format(index_trained.ntotal))
