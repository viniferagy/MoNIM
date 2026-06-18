import torch
import faiss
import math
import ctypes
import numpy as np
from numpy.lib.format import open_memmap
import os
from fairseq import utils, options
import time
from moe_modules import MLPCombiner, AdaptiveCombiner
import json

class KNN_Dstore(object):
    def __init__(self, args, tgt_dict):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        if args.continual and args.save_knnlm_dstore:
            a = json.load(open(f'{args.output_dir}/size.json', 'r'))
            *_, dataset, _, dstore_prefix = args.infer_dstore_path.split('/')
            info_list = dstore_prefix.split('_')
            ckpt_name, date, prune_param = info_list[0], info_list[1], info_list[-1]
            self.dstore_size = int(a[dataset][ckpt_name][date][prune_param])
            print('Previous size from json:', self.dstore_size)
        else:
            self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.drop_top1 = args.drop_top1
        self.knn_temp = args.knn_temp
        self.args = args
        self.prune_rate = args.prune_rate

        self.index = self.setup_faiss(args)

        self.model = None
        if args.ar_ckpt != 'none':
            self.model = self.setup_moe(args.ar_ckpt) # metak            
        
        self.dictionary = tgt_dict
        self.mask_for_label_count = None


    def setup_moe(self, ckpt_path):
        ckpt_moe = torch.load(ckpt_path, map_location=torch.device('cpu'))

        moe_args = ckpt_moe['args']
        moe_epoch = ckpt_moe['epoch']
        if 'mlp' in ckpt_path:
            moe_model = MLPCombiner(
                feature_size=moe_args.feature_size,
                hidden_units=moe_args.hidden_units,
                nlayers=moe_args.nlayers,
                dropout=moe_args.dropout,
                activation=moe_args.activation,
                )
        elif 'metak' in ckpt_path:
            moe_model = AdaptiveCombiner(
                max_k=1024,
                lambda_net_feature_size=moe_args.feature_size,
                lambda_net_hid_size=moe_args.hidden_units,
                lambda_net_nlayers=moe_args.nlayers,
                lambda_net_dropout_rate=moe_args.dropout,
                use_k=moe_args.use_k
            )
        moe_model.load_state_dict(ckpt_moe['param'])

        print(f'loaded models at epoch {moe_epoch} from {ckpt_path}')

        if torch.cuda.is_available():
            print('use cuda')
            # moe_model = moe_model.half()
            moe_model.cuda()

        moe_model.model.eval()

        return moe_model

    def setup_faiss(self, args):
        if not args.infer_dstore_path:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        if options.eval_bool(args.gpu_index):
            print('gpu faiss index')
            ngpus = faiss.get_num_gpus()
            print("number of GPUs:", ngpus)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self.faiss_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(self.faiss_res, 0, index, co)

        #     index.nprobe = args.probe

        fp = 16 if args.dstore_fp16 else 32
        print(f'Keys and vals are fp{fp}')
        if not args.no_load_keys:
            self.keys = open_memmap(args.infer_dstore_path+'_keys.npy', mode='r+')
        try:
            self.vals = open_memmap(args.infer_dstore_path+'_vals.npy', mode='r+')
        except:
            print('Old vals')
            self.vals = np.memmap(args.infer_dstore_path+'_vals.npy', mode='r+', dtype=np.int64, shape=(self.dstore_size, 1))
        # if have stored weight
        if os.path.exists(args.infer_dstore_path+'_weights.npy'):
            print(f'Use weight {args.infer_dstore_path}_weights.npy')
            self.weights = open_memmap(args.infer_dstore_path+'_weights.npy', mode='r+')
        else:
            print('No weight')
            self.weights = None

        if hasattr(self, 'keys'):
            # from https://github.com/numpy/numpy/issues/13172
            # to speed up access to memmap
            madvise = ctypes.CDLL("libc.so.6").madvise
            madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            madvise.restype = ctypes.c_int
            assert madvise(self.keys.ctypes.data, self.keys.size * self.keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = open_memmap(args.infer_dstore_path+'_keys.npy', mode='r+')
                self.keys = np.zeros((self.dstore_size, self.dimension), dtype=eval(f'np.float{fp}'))
                self.keys = self.keys_from_memmap[:]

            del self.vals
            self.vals_from_memmap = open_memmap(args.infer_dstore_path+'_vals.npy', mode='r+')
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int64)
            self.vals = self.vals_from_memmap[:]
            print('Loading to memory took {} s'.format(time.time() - start))

        return index


    def get_knns(self, queries):
        # import faiss.contrib.torch_utils后，search时可以直接使用tensor，不用转换为numpy，可以节省一点时间。但后面vals这些numpy数组使用切片都需要numpy index，所以还是保留
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)

        # test the overhead from data communication
        # queries = queries.detach().cpu().float().numpy()
        # bsz, feat = queries.shape
        # dists = np.random.rand(bsz, self.k)
        # knns = np.random.randint(103225480, size=(bsz, self.k))
        # knns = np.random.randint(19048862, size=(bsz, self.k))

        # TODO: this may be an ok way to avoid retrieving itself, but not guranteed due
        # to the aproximation, needs to be carefully checked
        if self.drop_top1:
            dists = dists[:, 1:]
            knns = knns[:, 1:]

        return dists, knns


    def get_knn_log_prob(self,
                         queries,
                         tgt,
                         pad_idx,
                         lmbda=None,
                         return_knn=False,
                         freq=None,
                         fert=None,
                         lm_entropy=None,
                         lm_max=None,
                         count_acc=False,
                         lm_scores=None):
        # print('get knn log prob')
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                # import pdb; pdb.set_trace()
                if self.metric_type == 'l2':
                    start = time.time()
                    # knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                    # import pdb; pdb.set_trace()
                    # if self.half:
                    #     knns_vecs = knns_vecs.half()
                    # query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    # l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)

                    # added by junxian
                    # perform distance recomputation on cpu to avoid gpu oom
                    knns_vecs = torch.from_numpy(self.keys[k]).view(qsize[0], self.k, -1)
                    # if self.half:
                    #     knns_vecs = knns_vecs.half()
                    query_vecs = q.cpu().view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum(((query_vecs - knns_vecs).float())**2, dim=2)
                    l2 = l2.cuda()
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                # (T_reducedxB)xKxC * (T_reducedxB)x1xC -> (T_reducedxB)xK
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # import pdb; pdb.set_trace()
        # queries  are TxBxC
        # reshape: (TxB)xC
        # import pdb; pdb.set_trace()
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        tgt = tgt.view(-1)
        knn_mask = (tgt != pad_idx)

        dists, knns = self.get_knns(queries[knn_mask])
        # print(f'retrieval consumes {time.time() - start} seconds')
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = dist_func(dists, knns, queries[knn_mask, :], function=self.sim_func)
        
        orig_dists = dists
        if self.weights is not None:
            # import pdb;pdb.set_trace()
            weights = self.weights[knns].squeeze(-1)
            weights_cumsum = np.cumsum(weights, axis=-1)
            # deal with boundary (actually has no impact on ppl!)
            # boundary = (weights_cumsum >= self.k).argmax(axis=-1)
            # weights[range(weights.shape[0]), boundary] = weights[range(weights.shape[0]), boundary] - (weights_cumsum[range(weights.shape[0]), boundary] - self.k)
            # weights_cumsum = np.cumsum(weights, axis=-1)
            weights[weights_cumsum > self.k] = 1e-100
            # dists here ↓ is dummy, only used to make weights on the same device 
            weights = dists.new_tensor(weights)
            dists = dists + torch.log(weights)
        # print(f'computing distance consumes {time.time() - start} seconds')
        # (T_reducedxB) x K
        vals = self.vals[knns].squeeze(-1)
        vals = dists.new_tensor(vals, dtype=torch.long).cuda()

        if self.model is not None:
            combine_scores, extra = self.model.get_prob({
                'val_id':vals, 
                'val_count':self._get_label_count_segment(vals),
                'old_distance':-orig_dists,
                'new_distance':-dists,
                'tgt':tgt[knn_mask],
                # lambda features
                'ctxt': queries[knn_mask].float(),
                'lm_ent': lm_entropy.view(-1, 1)[knn_mask].float() if lm_entropy is not None else None,
                'lm_max': lm_max.view(-1, 1)[knn_mask].float() if lm_max is not None else None,
                'freq': freq.view(-1, freq.size(-1))[knn_mask] if freq is not None else None,
                'fert': fert.view(-1, fert.size(-1))[knn_mask] if fert is not None else None,
                'knn_mask': knn_mask,
            }, 
            lm_scores=lm_scores.view(-1)[knn_mask] if lm_scores is not None else None,
            )
            knn_scores = extra['knn_scores']
            log_lambda = extra['log_lambda'].reshape(-1, 1)
            log_lambda = torch.concat([log_lambda, torch.log(torch.clip(1-torch.exp(log_lambda), min=1e-5))], dim=-1)
                
            self.knn_temp = extra['temp']
            probs = utils.log_softmax(dists / self.knn_temp, dim=-1)
        else:
            probs = utils.log_softmax(dists / self.knn_temp, dim=-1)
            index_mask = torch.eq(vals, tgt[knn_mask].unsqueeze(-1)).float()
            index_mask[index_mask == 0] = -10000 # for stability
            index_mask[index_mask == 1] = 0
            
            # (T_reducedxB)
            knn_scores = torch.logsumexp(probs + index_mask, dim=-1).clone()
            lm_scores = lm_scores.view(-1)[knn_mask]
            combine_scores = torch.stack([lm_scores, knn_scores], dim=-1)
            log_lambda = torch.ones_like(combine_scores)
            log_lambda[:, 0] = np.log(1 - lmbda)
            log_lambda[:, 1] = np.log(lmbda)
            combine_scores = torch.logsumexp(combine_scores + log_lambda, dim=-1)
        
        full_combine_scores = None
        if combine_scores is not None:
            full_combine_scores = dists.new_full([qshape[0]*qshape[1]], -10000)
            full_combine_scores[knn_mask] = combine_scores
            full_combine_scores = full_combine_scores.view(qshape[0], qshape[1])
        
        full_knn_probs = None
        if count_acc:
            # ------ count accuracy ------
            # (T_reducedxB) x vocab_size
            vocab_size = len(self.dictionary)
            knn_probs = torch.log(probs.new_zeros((vals.shape[0], vocab_size)).scatter_add_(-1, vals, torch.exp(probs)))
            knn_probs = torch.where(torch.isinf(knn_probs), torch.full_like(knn_probs, -10000), knn_probs)
            
            full_knn_probs = torch.full((qshape[0]*qshape[1],vocab_size), -10000, dtype=knn_probs.dtype).cuda()
            full_knn_probs[knn_mask] = knn_probs
            full_knn_probs = full_knn_probs.view(qshape[0],qshape[1],vocab_size)
            # ------ count accuracy ------
        full_knn_scores = torch.full([qshape[0]*qshape[1]], -10000, dtype=knn_scores.dtype).cuda()
        full_knn_scores[knn_mask] = knn_scores

        full_log_lambda, full_retrieval_mask = None, None
        if log_lambda is not None:
            full_log_lambda = torch.full([qshape[0]*qshape[1], 2], -10000, dtype=log_lambda.dtype).cuda()
            full_log_lambda[knn_mask, :] = log_lambda.view(-1, 2)
            full_log_lambda = full_log_lambda.view(qshape[0], qshape[1], 2)
            full_retrieval_mask = torch.full([qshape[0], qshape[1]], True, dtype=torch.float).cuda()
        # import pdb; pdb.set_trace()
        if return_knn:
            full_dists = dists.new_full([qshape[0]*qshape[1], orig_dists.size(-1)], -10000)
            full_dists[knn_mask] = -orig_dists
            # full_dists = full_dists[:, :10]

            new_dists = dists.new_full([qshape[0]*qshape[1], dists.size(-1)], -10000)
            new_dists[knn_mask] = -dists
            # new_dists = new_dists[:, :10]

            # knns = self.vals[knns[:, :10]].squeeze(-1)
            full_knns = dists.new_full([qshape[0]*qshape[1], knns.shape[1]], -10000, dtype=torch.int)
            full_knns[knn_mask] = dists.new_tensor(knns, dtype=torch.int)
            
            full_vals = dists.new_full([qshape[0]*qshape[1], vals.shape[1]], -10000, dtype=torch.long)
            full_vals[knn_mask] = vals
            full_vals = full_vals.reshape(qshape[0], qshape[1], -1)

            val_counts = self._get_label_count_segment(full_vals)
            # import pdb; pdb.set_trace()
            return full_knn_scores.view(qshape[0], qshape[1], 1), full_dists.view(qshape[0], qshape[1], -1), \
                    new_dists.view(qshape[0], qshape[1], -1), full_knns.view(qshape[0], qshape[1], -1), full_vals, val_counts, full_log_lambda, full_retrieval_mask, full_knn_probs, full_combine_scores

        else:
            # return dists for analysis purpose
            # TxBx1
            return full_knn_scores.view(qshape[0], qshape[1], 1), None, None, None, None, None, full_log_lambda, full_retrieval_mask, full_knn_probs, full_combine_scores

    def _get_label_count_segment(self, vals, relative=False):
            r""" this function return the label counts for different range of k nearest neighbor 
                [[0:0], [0:1], [0:2], ..., ]
            """
            
            # caculate `label_count_mask` only once
            if self.mask_for_label_count is None:
                mask_for_label_count = torch.empty((self.k, self.k)).fill_(1)
                mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()
                mask_for_label_count.requires_grad = False
                # [0,1,1]
                # [0,0,1]
                # [0,0,0]
                self.mask_for_label_count = mask_for_label_count.to(vals.device)

            ## TODO: The feature below may be unreasonable
            vals_2d = vals.reshape(-1, vals.shape[-1])
            B, K = vals_2d.shape
            retrieve_label_counts = torch.full_like(vals_2d, 1)
            for i in range(B):
                if vals_2d[i, 0] == -10000:
                    continue
                expand_vals = vals_2d[i, :].unsqueeze(-2).expand(1,K,K)
                expand_vals = expand_vals.masked_fill(self.mask_for_label_count, value=-1)
                

                labels_sorted, _ = expand_vals.sort(dim=-1) # [B*S, K, K]
                labels_sorted[:, :, 1:] *= ((labels_sorted[:, :, 1:] - labels_sorted[:, : , :-1]) != 0).long()
                retrieve_label_count = labels_sorted.ne(0).sum(-1)
                retrieve_label_count[:, :-1] -= 1
                retrieve_label_counts[i, :] = retrieve_label_count

            # if relative:
            #     relative_label_counts[:, :, 1:] = relative_label_counts[:, :, 1:] - relative_label_counts[:, :, :-1]
            retrieve_label_counts = retrieve_label_counts.reshape_as(vals)
            return retrieve_label_counts
