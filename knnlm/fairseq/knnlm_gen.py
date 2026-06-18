import torch
import faiss
import math
import ctypes
import numpy as np

from torch.cuda.amp import autocast

from fairseq import utils, options
import time
from fairseq.data import Dictionary

from moe_modules import MLPMOE


class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.dimension
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.drop_top1 = args.drop_top1
        self.knn_temp = args.knn_temp
        self.args = args

        self.index = self.setup_faiss(args)

        if args.ar_ckpt != '' and args.ar_ckpt != 'none':
            self.moe = self.setup_moe(args.ar_ckpt, args.ar_cutoff)
        else:
            self.moe = None
        
        from fairseq.data import Dictionary
        self.dictionary = Dictionary.load(f'{args.data}/dict.txt')

    def setup_moe(self, ckpt_path, ar_cutoff=50):
        ckpt_moe = torch.load(ckpt_path)

        moe_args = ckpt_moe['args']
        moe_epoch = ckpt_moe['epoch']
        self.moe_threshold = ckpt_moe['threshold'][ar_cutoff]
        # self.moe_threshold=999
        moe_model = MLPMOE(
            feature_size=moe_args.feature_size,
            hidden_units=moe_args.hidden_units,
            nlayers=moe_args.nlayers,
            dropout=moe_args.dropout,
            activation=moe_args.activation,
            )

        moe_model.load_state_dict(ckpt_moe['param'])

        print(f'loaded models at epoch {moe_epoch} from {ckpt_path}')
        print(f'cutoff {ar_cutoff}, threshod: {self.moe_threshold}')

        if torch.cuda.is_available():
            print('use cuda')
            # moe_model = moe_model.half()
            moe_model.cuda()

        moe_model.eval()

        return moe_model

    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        if options.eval_bool(args.gpu_index):
            print('gpu faiss index')
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index, co)

            index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=int, mode='r', shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=int, mode='r', shape=(self.dstore_size, 1))

        if args.prune:
            print('Use weight')
            self.weights = np.memmap(args.dstore_filename+'_weights.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, 1))
        else:
            self.weights = None

        if hasattr(self, 'keys'):
            # from https://github.com/numpy/numpy/issues/13172
            # to speed up access to np.memmap
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
                dtype_keys = np.float16 if args.dstore_fp16 else np.float32
                self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=dtype_keys, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension), dtype=dtype_keys)
                self.keys = self.keys_from_memmap[:]

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=int, mode='r', shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=int)
            self.vals = self.vals_from_memmap[:]
            print('Loading to memory took {} s'.format(time.time() - start))

        return index


    def get_knns(self, queries):
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
                         queries):
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
        # queries are (bsz*beam_size) x dim
        # import pdb; pdb.set_trace()
        # tgt = tgt.contiguous().view(-1)
        log_moe_lmw = retrieval_mask =  None

        dists, knns = self.get_knns(queries)
        # print(f'retrieval consumes {time.time() - start} seconds')
        # (bsz*beam_size) x K
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = dist_func(dists, knns, queries, function=self.sim_func)
        
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
        probs = utils.log_softmax(dists / self.knn_temp, dim=-1)
        # (bsz*beam_size) x K
        token_index = torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1)
        
        # print(token_index.shape)
        # print(self.dictionary.string(token_index[:,:100]))

        # (bsz*beam_size) * vocab_size
        knn_probs = torch.log(probs.new_zeros((token_index.shape[0], 50261)).scatter_add_(-1, token_index, torch.exp(probs)))
        knn_probs = torch.where(torch.isinf(knn_probs), torch.full_like(knn_probs, -10000), knn_probs)
        # print(token_index)
        # print(token_index.shape)
        # print(probs)
        # print(probs.shape)
        # print(knn_probs[0, :])
        # 1/0
        # (T_reducedxB)
        # yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()
        # full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000, dtype=yhat_knn_prob.dtype).cuda()
        # full_yhat_knn_prob[knn_mask] = yhat_knn_prob
        # if use_adaptive_weight:
        #     full_adaptive_weight = torch.full([qshape[0] * qshape[1]], 1e-5, dtype=yhat_knn_prob.dtype).cuda()
        #     full_adaptive_weight[knn_mask] = adaptive_weight
        #     full_adaptive_weight = full_adaptive_weight.view(qshape[0], qshape[1])
        # else:
        #     full_adaptive_weight = None

        # if log_moe_lmw is not None:
        #     log_moe_lmw = log_moe_lmw.view(qshape[0], qshape[1])
        #     retrieval_mask = retrieval_mask.view(qshape[0], qshape[1]).float()

        # import pdb; pdb.set_trace()
        # if return_knn:
        #     full_dists = dists.new_full([qshape[0]*qshape[1], orig_dists.size(-1)], -10000)
        #     full_dists[knn_mask] = -orig_dists
        #     full_dists = full_dists[:, :10]

        #     new_dists = dists.new_full([qshape[0]*qshape[1], dists.size(-1)], -10000)
        #     new_dists[knn_mask] = -dists
        #     new_dists = new_dists[:, :10]

        #     knns = self.vals[knns[:, :10]].squeeze(-1)
        #     full_knns = dists.new_full([qshape[0]*qshape[1], knns.shape[1]], -10000, dtype=torch.int)
        #     full_knns[knn_mask] = dists.new_tensor(knns, dtype=torch.int)

        #     # import pdb; pdb.set_trace()
        #     return full_yhat_knn_prob.view(qshape[0], qshape[1], 1), log_moe_lmw, retrieval_mask, full_dists.view(qshape[0], qshape[1], -1), \
        #             new_dists.view(qshape[0], qshape[1], -1), full_knns.view(qshape[0], qshape[1], -1), full_adaptive_weight

        # else:
            # return dists for analysis purpose
            # TxBx1
        # return full_yhat_knn_prob.view(qshape[0], qshape[1], 1), log_moe_lmw, retrieval_mask, None, None, None, full_adaptive_weight
        return knn_probs

