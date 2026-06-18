import json
import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm, trange

from collections import Counter, OrderedDict
from tqdm import tqdm
from moe_modules import MLPCombiner, LSTMMOE, TokenFeatureDataset, AdaptiveCombiner
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

rank = int(os.environ["RANK"])
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ["LOCAL_RANK"])
p_pace = 10 # progress bar update frequency

class Logger(object):
  def __init__(self, output_file, rank):
    self.terminal = sys.stdout
    self.log = open(output_file, "w+")
    self.rank = rank

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()


def validate(val_dataloader, model, args):
    model.module.model.eval()
    # model.model.epoch_update()
    running_loss = torch.tensor(0., device=gpu)
    nsamples = 0
    for i, sample in enumerate(val_dataloader, 0):
        inputs, lm_scores = sample['feature'], sample['lm_scores']
        loss, extra = model.module.get_loss(inputs, lm_scores, device="cuda:0", l1=args.l1)
        bsz = next(iter(inputs.values())).size(0)

        running_loss += loss * bsz
        nsamples += bsz
    
    val_loss = running_loss / nsamples
    dist.barrier()
    reduced_loss = reduce_tensor(val_loss)
    print(f"val loss: {reduced_loss:.3f}, ppl: {2**(reduced_loss)}")
    
    return reduced_loss

def read_input(input, debug=False):
    input_list = []
    for i in range(world_size):
        input_list.append(f'{input}.{i}.jsonl')
    print('Reading', input)
    dataset = datasets.load_dataset('json', data_files=input_list, cache_dir=args.cache_path, keep_in_memory=True)

    return dataset['train']

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

parser = argparse.ArgumentParser(description='')

parser.add_argument('--train', type=str, default=None,
    help='the input feature file (jsonl)')
parser.add_argument('--val', type=str, default=None,
    help='the input feature file (jsonl)')
parser.add_argument('--train-others', type=str, default=None,
    help='use a specified jsonl file for others feature if specified')
parser.add_argument('--val-others', type=str, default=None,
    help='use a specified jsonl file for others feature if specified')
parser.add_argument('--input', type=str, default=None,
    help='the input feature file (jsonl). Multiple files are separated with comma')
parser.add_argument('--negative-weight', type=float, default=1,
        help='weight of the loss from negative examples, range [0,1]')
parser.add_argument('--feature-type', type=str, default='all',
    help='the features to use, splitted with commas')
parser.add_argument('--seed', type=int, default=22,
    help='the random seed')
parser.add_argument('--debug', action='store_true', default=False,
    help='debug mode')

# training arguments
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--l1', type=float, default=0.,
    help='l1 regularization coefficient')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--ngram', type=int, default=0, help='the ngram features to use')

# model hyperparameters
parser.add_argument('--arch', type=str, choices=['mlp', 'lstm', 'metak'], default='mlp',
    help='architectures of the expert model')
parser.add_argument('--activation', type=str, choices=['linear', 'relu'], default='relu',
    help='the activation function in mlp')
parser.add_argument('--hidden-units', type=int, default=32, help='hidden units')
parser.add_argument('--nlayers', type=int, default=3, help='number of layerss')
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--k', type=int, default=1024, help='dropout')
parser.add_argument('--dict_', type=float, default=0, help='dropout')

parser.add_argument('--output-dir', type=str)
parser.add_argument('--move-to-mem', action='store_true', default=False)
parser.add_argument('--load-model', type=str, default=None,
    help='load model checkpoint')
parser.add_argument('--eval', action='store_true', default=False,
    help='perform evaluation')
parser.add_argument('--save-pred', type=str, default=None,
    help='save predictions for analysis')
parser.add_argument('--validate-loss', action='store_true', default=False,
    help='save predictions for analysis')

parser.add_argument('--use-k', type=int, default=None)
parser.add_argument('--cache-path', type=str, default='hf_cache')

args = parser.parse_args()

gpu = local_rank

logfile = 'stdout.log' if not args.eval else 'eval.log'
sys.stdout = Logger(os.path.join(args.output_dir, logfile), gpu) if gpu == 0 else None

print(args)

############################################################
dist.init_process_group(                                    
    backend='nccl',                                         
    init_method='env://',                                    
    world_size=world_size,                      
    rank=rank                                                
)                                                          
############################################################

np.random.seed(args.seed+gpu)
torch.manual_seed(args.seed+gpu)
torch.cuda.set_device(gpu)

batch_size = int(args.batch_size / world_size)

train_ctxt_hypos = read_input(args.train + '_ctxt', debug=args.debug)
if args.train_others is None:
    train_other_hypos = read_input(args.train + '_others', debug=args.debug)
else:
    train_other_hypos = read_input(args.train_others)

val_ctxt_hypos = read_input(args.val + '_ctxt', debug=args.debug)
if args.val_others is None:
    val_other_hypos = read_input(args.val + '_others', debug=args.debug)
else:
    val_ctxt_hypos = read_input(args.val_others)

if args.move_to_mem:
    train_ctxt_hypos = [train_ctxt_hypos[i] for i in range(len(train_ctxt_hypos))]
    train_other_hypos = [train_other_hypos[i] for i in range(len(train_other_hypos))]
    val_ctxt_hypos = [val_ctxt_hypos[i] for i in range(len(val_ctxt_hypos))]
    val_other_hypos = [val_other_hypos[i] for i in range(len(val_other_hypos))]

print('complete reading jsonl files')

training_set = TokenFeatureDataset(train_ctxt_hypos, train_other_hypos, ngram=args.ngram)
val_set = TokenFeatureDataset(val_ctxt_hypos, val_other_hypos, ngram=args.ngram)

################################################################
# By default, :attr:`world_size` and `rank` is retrieved from the current distributed group.
train_sampler = DistributedSampler(training_set)
valid_sampler = DistributedSampler(val_set)
################################################################

train_dataloader = DataLoader(training_set,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=train_sampler,
                            collate_fn=training_set.collater)

val_dataloader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=valid_sampler,
                            collate_fn=val_set.collater)

step_per_gpu = len(train_dataloader)
total_step = step_per_gpu
print(total_step)
if total_step < 500:
    nepochs = 10
elif total_step < 2000:
    nepochs = 7
elif total_step < 5000:
    nepochs = 5
elif total_step < 10000:
    nepochs = 3
else:
    nepochs = 2
# nepochs = 10
extra_feature_size = None

feature_set = ['ctxt', 'freq', 'lm_ent', 'lm_max', 'fert']

# if args.feature_type == 'all':
feature_size = OrderedDict({key: training_set.get_nfeature(key) for key in feature_set})
# else:
#     feature_size = OrderedDict({key: training_set.get_nfeature(key) for key in args.feature_type.split(',')})

args.feature_size = feature_size

if args.arch == 'mlp':
    model = MLPCombiner(
                feature_size=feature_size,
                hidden_units=args.hidden_units,
                nlayers=args.nlayers,
                dropout=args.dropout,
                activation=args.activation,
                )
elif args.arch == 'metak':
    model = AdaptiveCombiner(
        max_k=1024,
        lambda_net_feature_size=feature_size,
        lambda_net_hid_size=args.hidden_units,
        lambda_net_nlayers=args.nlayers,
        lambda_net_dropout_rate=args.dropout,
        use_k=args.use_k,
    )
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([args.negative_weight, 1]))

if args.load_model:
    ckpt_path = os.path.join(args.load_model, 'checkpoint_best.pt')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['param'])
    print(f"loaded model ckpt from {ckpt_path} at epoch {ckpt['epoch']}")
    val_loss = validate(val_dataloader, model, args)
    exit()
model.cuda(gpu)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
###############################################################
# Wrap the model
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[gpu])
###############################################################
# print(model)

model.module.model.train()
best_loss = 1e5
ret = torch.tensor([0]).cuda(gpu)
for epoch in range(nepochs):
    if gpu == 0: 
        pbar = trange(step_per_gpu)
    running_loss = torch.tensor(0., device=gpu)
    nsamples = 0
    # Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch. Otherwise, the next iteration of this sampler will yield the same ordering.
    train_dataloader.sampler.set_epoch(epoch)
    for i, sample in enumerate(train_dataloader):
        inputs, lm_scores = sample['feature'], sample['lm_scores']
        loss, extra = model.module.get_loss(inputs, lm_scores, device=gpu, l1=args.l1)
        optimizer.zero_grad()
        loss.backward()

        if torch.isnan(loss).any():
            args.lr = args.lr / 2
            print(f'{gpu}: Lr too big. Reset lr to {args.lr}')
            print(f'{gpu} exit')
            exit(-1)
        
        optimizer.step()

        bsz = next(iter(inputs.values())).size(0)
        running_loss += loss * bsz
        nsamples += bsz
        if i % p_pace == 0: # update progress bar
            report_loss = running_loss / nsamples
            dist.barrier()
            reduced_loss = reduce_tensor(report_loss)
            if gpu == 0:
                pbar.set_description(f'Epoch [{epoch + 1}/{nepochs}], Step [{i + 1}/{total_step}], Loss: {report_loss:.4f}-{reduced_loss:.4f}, ppl: {2**(reduced_loss):.4f}, lr: {args.lr}')
                pbar.update(p_pace)
        
        if (i+1) % 3000 == 0:
            val_loss = validate(val_dataloader, model, args)
            if val_loss < best_loss:
                best_loss = val_loss
                if gpu == 0:
                    print('save model')
                    torch.save({'epoch': epoch,
                            'args': args,
                            'param': model.module.state_dict()},
                            os.path.join(args.output_dir, 'checkpoint_best.pt'))
            model.module.model.train()
    if gpu == 0: 
        pbar.close()
    val_loss = validate(val_dataloader, model, args)
    if val_loss < best_loss:
        best_loss = val_loss
        if gpu == 0:
            print('save model')
            torch.save({'epoch': epoch,
                    'args': args,
                    'param': model.module.state_dict()},
                    os.path.join(args.output_dir, 'checkpoint_best.pt')) 
    model.module.model.train()
print(f'best val ppl: {2**(best_loss):.3f}')
