import argparse
import os
import numpy as np
from numpy.lib.format import open_memmap
import time
import json
from tqdm import trange, tqdm
import itertools
import fcntl

def tryLock(f) :
    try :
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except Exception as e:
        return False

def tryUnLock(f) :
    try :
        fcntl.flock(f, fcntl.LOCK_UN)
        return True
    except Exception as e:
        return False

MAX_DSTORE_SIZE = 300000000

parser = argparse.ArgumentParser()

parser.add_argument('--old-dstore', type=str, default=None)
parser.add_argument('--new-dstore', type=str, default=None)
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore-fp16', default=False, action='store_true')
parser.add_argument('--output-dir', type=str, default=None)
parser.add_argument('--num-shards', type=int, default=None)

args = parser.parse_args()
print(args)

start = time.time()
*_, dataset, _, dstore_prefix = args.new_dstore.split('/')    
info_list = dstore_prefix.split('_')
ckpt_name, new_date, prune_param = info_list[0], info_list[1], info_list[-1]
a = json.load(open(f'{args.output_dir}/size.json', 'r'))
fp = 16 if args.dstore_fp16 else 32

new_size = 0
new_total_size = a[dataset][ckpt_name][new_date]['total']

# useful_counts = 0
# for i in range(args.num_shards):
#     useful_counts += int(open(f'useful{i}.tmp', 'r').read())
# print(f'Useful: {useful_counts}/{new_total_size}-{useful_counts/new_total_size:.4f}')

new_keys_list = []
new_vals_list = []

print('Loading new dstores...', end='')
for i in range(args.num_shards):
    if os.path.exists(args.new_dstore+f'_keys.today.{i}.npy'):
        new_keys = np.load(args.new_dstore+f'_keys.today.{i}.npy', mmap_mode='r')
        new_vals = np.load(args.new_dstore+f'_vals.today.{i}.npy', mmap_mode='r')
        new_size += new_keys.shape[0]
        new_keys_list.append(new_keys)
        new_vals_list.append(new_vals)
        print(f'{i}...', end='')
    else:
        break
print('Done')
print(f'New: {new_size}({new_size/new_total_size:.4f})')

# read old_size, new_size from size.json
if args.old_dstore:
    *_, old_dataset, _, dstore_prefix = args.old_dstore.split('/')
    info_list = dstore_prefix.split('_')
    ckpt_name, old_date, prune_param = info_list[0], info_list[1], info_list[-1]
    
    old_size = int(a[old_dataset][ckpt_name][old_date][prune_param])
    old_keys = open_memmap(args.old_dstore+'_keys.npy', mode='r+')
    old_vals = open_memmap(args.old_dstore+'_vals.npy', mode='r+')
else: # first update
    old_date = 0
    old_size = 0
    old_keys = open_memmap(args.new_dstore+'_keys.npy', dtype=eval(f'np.float{fp}'), mode='w+', shape=(MAX_DSTORE_SIZE, args.dimension))
    old_vals = np.array([]).reshape(-1, 1)

merged_size = old_size + new_size
print(f'Merge old: {old_size} + new: {new_size}({new_size/new_total_size:.4f}) = {merged_size} instances in dstore')

merged_vals = open_memmap(args.new_dstore+'_vals.npy', dtype=np.int64, mode='w+', shape=(merged_size, 1))
merged_vals[:old_size] = old_vals

for (k, v) in zip(new_keys_list, new_vals_list):
    old_keys[old_size:old_size+k.shape[0]] = k
    merged_vals[old_size:old_size+v.shape[0]] = v
    old_size += k.shape[0]

old_keys.flush()
merged_vals.flush()

if args.old_dstore:
    os.rename(args.old_dstore+'_keys.npy', args.new_dstore+'_keys.npy')
print(f'Merged in {time.time()-start} s')

# write merged_size to size.json
f = open(f'{args.output_dir}/size.json', 'r')
while tryLock(f) == False:
    print('Waiting for lock...')
    time.sleep(1)
print('Locked size.json...', end='')
a = json.load(f)
a[dataset][ckpt_name][new_date][prune_param] = merged_size
a[dataset][ckpt_name][new_date][prune_param+'-now'] = new_size
# a[dataset][ckpt_name][new_date][prune_param+'-in-1.5'] = useful_counts
json.dump(a, open(f'{args.output_dir}/size.json', 'w+'), indent=4)
f.close()
print('Unlocked size.json')