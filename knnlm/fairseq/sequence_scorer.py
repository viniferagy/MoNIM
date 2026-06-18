# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys
import numpy as np
import time

from fairseq import utils
from fairseq.data import Dictionary


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False, args=None):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.args = args
        self.tp_stop = False
        from fairseq.data import Dictionary
        self.dictionary = tgt_dict

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        # import pdb; pdb.set_trace()
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        def combine_lm_and_knn_probs(lm_p, knn_p, coeff=None, log_lambda=None, retrieval_mask=None, adaptive_weight=None):
            if adaptive_weight is not None:
                if retrieval_mask is not None:
                    knn_weights = adaptive_weight * retrieval_mask
                else:
                    knn_weights = adaptive_weight
                knn_weights = knn_weights.clone().detach().cpu()
                # knn_p[knn_p < -10] = -10
                combine_probs = torch.stack([lm_p, knn_p], dim=0)
                coeffs = torch.ones_like(combine_probs)
                coeffs[0] = np.log(np.clip(1 - knn_weights, 1e-5, 1))
                coeffs[1] = np.log(np.clip(knn_weights, 1e-5, 1))
                curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
            elif log_lambda is None:
                combine_probs = torch.stack([lm_p, knn_p], dim=0)
                coeffs = torch.ones_like(combine_probs)
                coeffs[0] = np.log(1 - coeff)
                coeffs[1] = np.log(coeff)
                curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
            else:
                # use learned mixture of experts to perform interpolation
                # log_moe_w = torch.full_like(log_moe_w, np.log(1 - coeff))
                # lm_weights = retrieval_mask * log_lambda
                # knn_prob = 1. - torch.exp(log_lambda)
                # overflow = (knn_prob <= 0)
                # knn_prob = torch.clip(knn_prob, 1e-5, 1)
                # knn_weights = torch.log(knn_prob)
                # knn_weights[overflow] = -1e5
                # knn_weights = retrieval_mask * knn_weights + (1-retrieval_mask) * (-1e5)

                # combine_probs = torch.stack([vocab_p + lm_weights, knn_p + knn_weights], dim=0)
                # curr_prob = torch.logsumexp(combine_probs, dim=0)
                curr_prob = log_lambda.unsqueeze(-2) + torch.stack((lm_p, knn_p), dim=-1)
                curr_prob = torch.logsumexp(curr_prob, dim=-1)

            return curr_prob

        # [bsz, max_tokens]
        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        cur_time = time.time()
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            # print(decoder_out)
            # print(decoder_out[0].shape)
            # raise AssertionError
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, lm_probs, idx = None, None, 0
            lm_vocab_probs, knn_vocab_probs = None, None
            net_lambda = None
            lm_entropy = None
            lm_max = None

            lm_entropy_flag = lm_max_flag = False
            if self.args and self.args.save_feature is not None:
                lm_entropy_flag = lm_max_flag = True
            elif self.args.ar_ckpt != 'none':
                if 'lm_ent' in self.args.ar_feat_type:
                    lm_entropy_flag = True

                if 'lm_max' in self.args.ar_feat_type:
                    lm_max_flag = True

            retrieval_mask_1 = None
            stop_rank = []
            nwp_correct = []
            vars = []
            for piece, (bd, tgt, is_single) in enumerate(batched):
                # print(bd[0].shape)
                sample['target'] = tgt
                if 'freq' in sample:
                    freq = sample['freq'].reshape(-1, sample['freq'].shape[-1])
                if 'fert' in sample:
                    fert = sample['fert'].reshape(-1, sample['fert'].shape[-1])
                # to have log prob for every word (in the adaptive softmax case)
                # [1, 1024, 267744]
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=None).data
                # import pdb; pdb.set_trace()
                # if is_single:
                #     print(curr_prob.shape, orig_target.shape)
                #     lm_probs = gather_target_probs(curr_prob, orig_target)

                #     # currently entropy computation only supports single model
                #     if lm_entropy_flag:
                #         # import pdb; pdb.set_trace()
                #         lm_entropy = -(curr_prob.exp() * curr_prob).sum(dim=-1)

                #     if lm_max_flag:
                #         lm_max, _ = curr_prob.max(dim=-1)
                if True:
                    if True:
                        lm_vocab_probs = curr_prob.new_zeros((orig_target.numel(), curr_prob.shape[-1]))
                        knn_vocab_probs = curr_prob.new_zeros((orig_target.numel(), curr_prob.shape[-1]))
                        net_lambda = curr_prob.new_full((orig_target.numel(), 2), -10000)
                    if lm_probs is None:
                        lm_probs = curr_prob.new(orig_target.numel())
                        if lm_entropy_flag:
                            lm_entropy = curr_prob.new(orig_target.numel())

                        if lm_max_flag:
                            lm_max = curr_prob.new(orig_target.numel())

                        # entropy todo here
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx

                    if lm_vocab_probs is not None:
                        lm_vocab_probs[idx:end] = curr_prob.view(-1, curr_prob.shape[-1])

                    if lm_entropy_flag:
                        # entropy_local = curr_prob.view(tgt.shape + (curr_prob.size(-1),))
                        # entropy_local = -(entropy_local.exp() * entropy_local).sum(dim=-1)
                        entropy_local = -(curr_prob.exp() * curr_prob).sum(dim=-1)
                        lm_entropy[idx:end] = entropy_local.view(-1)
                        # pass

                    if lm_max_flag:
                        # max_prob, _ = curr_prob.view(tgt.shape + (curr_prob.size(-1),)).max(dim=-1)
                        max_prob, _ = curr_prob.max(dim=-1)
                        lm_max[idx:end] = max_prob.view(-1)

                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    lm_probs[idx:end] = tgt_probs.view(-1)

                    if self.args.count_acc:
                        if nwp_correct == []:
                            # nwp_answer = curr_prob.new(orig_target.numel())
                            nwp_correct = curr_prob.new(orig_target.numel())
                            if self.args.deviation:
                                vars = curr_prob.new(orig_target.numel())
                        if 'knn_dstore' in kwargs:
                            if probs is None: # 顺便把probs也计算了
                                probs = curr_prob.new(orig_target.numel())
                            dstore = kwargs['knn_dstore']
                            # TxBxC
                            queries_shape = bd[1][self.args.knn_keytype].shape
                            queries = bd[1][self.args.knn_keytype].permute(1,0,2).contiguous().reshape(-1, queries_shape[-1])
                            queries_piece = queries[idx:end].unsqueeze(0)
                            *_, log_lambda, retrieval_mask, knn_probs_piece, probs_piece = dstore.get_knn_log_prob(
                                    queries_piece,
                                    tgt,
                                    pad_idx=self.pad,
                                    lmbda=self.args.lmbda,
                                    return_knn=self.args.analyze_knn,
                                    count_acc=True,
                                    freq=freq[idx:end] if 'freq' in sample else None,
                                    fert=fert[idx:end] if 'fert' in sample else None,
                                    lm_entropy=lm_entropy[idx:end] if lm_entropy is not None else None,
                                    lm_max=lm_max[idx:end].exp() if lm_max is not None else None,
                                    lm_scores=lm_probs[idx:end],
                                    )
                            probs[idx:end] = probs_piece.view(-1)
                            if self.args.end_task:
                                knn_vocab_probs[idx:end] = knn_probs_piece.view(-1, curr_prob.shape[-1])
                                net_lambda[idx:end] = log_lambda.view(-1, 2)
                            # print(knn_probs_piece.shape)
                            # raise AssertionError
                            # nwp_probs_piece = combine_lm_and_knn_probs(curr_prob, knn_probs_piece, log_lambda=log_lambda)
                            knn_probs_piece = knn_probs_piece.view((*tgt.shape, knn_probs_piece.shape[-1]))
                            nwp_probs_piece = log_lambda.unsqueeze(-2) + torch.stack((curr_prob, knn_probs_piece), dim=-1)
                            nwp_probs_piece = torch.logsumexp(nwp_probs_piece, dim=-1)
                            # print(nwp_answers_piece)
                            # print(nwp_answers_piece.shape)
                            # raise AssertionError
                        else: # w/o knn
                            # [1, softmax_batch, vocab_size]
                            nwp_probs_piece = curr_prob
                        # tgt not all 1's
                        if not torch.eq(tgt, torch.full_like(tgt, fill_value=1)).all():
                            # tmp = torch.argmax(nwp_probs_piece, dim=-1)
                            tmp = torch.argsort(nwp_probs_piece, dim=-1, descending=True)[:, :, :1]
                            # nwp_correct_pixel = torch.eq(tmp, tgt)
                            nwp_correct_pixel = torch.eq(tmp, tgt[..., None]).any(dim=-1)
                            nwp_correct[idx:end] = nwp_correct_pixel
                            if self.args.deviation:
                                var_piece = torch.var(nwp_probs_piece, dim=-1)
                                vars[idx:end] = var_piece

                    
                    idx = end
                    # --------------- Melongena hacking (stop words not use knn) ------------
                    if self.tp_stop:
                        may_be_stop = []
                        ir = []
                        for i in range(0, 1024, 128):
                            tmp = torch.argsort(curr_prob[:, i:i+128, :], dim=-1, descending=True)
                            index_rank = tmp == tgt[:, i:i+128, None]
                            index_rank = torch.argmax(index_rank * 1, dim=-1)
                            ir.append(index_rank)
                            index_rank = tmp[..., :5]
                            may_be_stop_i = torch.isin(index_rank, self.args.stop).sum(dim=-1) >= 3
                            may_be_stop.append(may_be_stop_i)
                        may_be_stop = torch.concat(may_be_stop, dim=-1).squeeze()
                        ir = torch.concat(ir, dim=-1).squeeze()
                        mask = torch.nonzero(torch.isin(tgt.squeeze(), self.args.stop)).squeeze()
                        ir = ir[mask]
                        stop_rank.append(ir)
                        if retrieval_mask_1 is None:
                            retrieval_mask_1 = ~may_be_stop
                        else:
                            retrieval_mask_1 = torch.concat([retrieval_mask_1, ~may_be_stop])

                sample['target'] = orig_target

            # full_lm_probs = torch.concat(full_lm_probs, dims=1)
            # print(full_lm_probs.shape)

            lm_probs = lm_probs.view(sample['target'].shape)
            if retrieval_mask_1 is not None:
                retrieval_mask_1 = retrieval_mask_1.reshape_as(lm_probs)
                a = torch.sum(retrieval_mask_1)
                print(a)

            if lm_entropy is not None:
                lm_entropy = lm_entropy.view(sample['target'].shape)

            if lm_max is not None:
                lm_max = lm_max.view(sample['target'].shape)
            
            if nwp_correct != []:
                nwp_correct = nwp_correct.view(sample['target'].shape)
                if self.args.deviation:
                    vars = vars.view(sample['target'].shape)

            if lm_vocab_probs is not None:
                lm_vocab_probs = lm_vocab_probs.view((*sample['target'].shape, lm_vocab_probs.shape[-1]))
                knn_vocab_probs = knn_vocab_probs.view((*sample['target'].shape, lm_vocab_probs.shape[-1]))
                net_lambda = net_lambda.view((*sample['target'].shape, 2))
            # print(f'forward consumes {time.time() - cur_time} seconds')

            if self.args and self.args.save_feature is not None:
                lm_context = bd[1][self.args.knn_keytype].permute(1, 0, 2)

            else:
                lm_context = None
            if probs is not None:
                probs = probs.view(sample['target'].shape)
            else:
                if 'knn_dstore' in kwargs:
                    dstore = kwargs['knn_dstore']
                    # TxBxC 3072 x 6 x 1024
                    queries = bd[1][self.args.knn_keytype]
                    # if self.args.save_feature is not None:
                    #     lm_context = queries.permute(1, 0, 2)
                    # else:
                    #     lm_context = None
                    # import pdb; pdb.set_trace()
                    if len(models) != 1:
                        raise ValueError('Only knn *log* probs are supported.')
                    yhat_knn_prob, knn_old_dists, knn_new_dists, knn_ids, val_ids, val_counts, log_lambda, retrieval_mask, _, probs = dstore.get_knn_log_prob(
                            queries.permute(1,0,2).contiguous(),
                            orig_target,
                            pad_idx=self.pad,
                            lmbda=self.args.lmbda,
                            return_knn=self.args.analyze_knn,
                            freq=sample['freq'] if 'freq' in sample else None,
                            fert=sample['fert'] if 'fert' in sample else None,
                            lm_entropy=lm_entropy if lm_entropy is not None else None,
                            lm_max=lm_max.exp() if lm_max is not None else None,
                            lm_scores=lm_probs,
                            )
                    # yhat_knn_prob = yhat_knn_prob.permute(1, 0, 2).squeeze(-1)
                    yhat_knn_prob = yhat_knn_prob.squeeze(-1)
                    if self.args.fp16:
                        yhat_knn_prob = yhat_knn_prob.half()
                        lm_probs = lm_probs.half()

                    # print(f'knn consumes {time.time() - cur_time} seconds')

                    # cur_time = time.time()
                    # if probs is None:
                    #     if self.args.stop_size != '':
                    #         retrieval_mask = torch.isin(orig_target, self.args.stop, invert=True)
                    #     if self.tp_stop:
                    #         retrieval_mask = retrieval_mask_1
                    #     probs = combine_lm_and_knn_probs(
                    #                 lm_probs, yhat_knn_prob,
                    #                 self.args.lmbda, log_lambda,
                    #                 retrieval_mask, 
                    #                 adaptive_weight)
                    # print(f'interpolation consumes {time.time() - cur_time} seconds')


            if avg_probs is None:
                if probs is not None:
                    avg_probs = probs
                else:
                    avg_probs = lm_probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        cur_time = time.time()
        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        lm_probs_i = lm_entropy_i = lm_max_i = knn_probs_i = lm_context_i = None
        knn_old_dists_i = knn_new_dists_i = knn_ids_i = val_ids_i = val_count_i = None
        # import pdb; pdb.set_trace()
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            if self.args and self.args.save_feature is not None:
                lm_probs_i = lm_probs[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
                lm_entropy_i = lm_entropy[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
                lm_max_i = lm_max[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
                lm_context_i = lm_context[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()

                if self.args.analyze_knn:
                    knn_old_dists_i = knn_old_dists[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()
                    # knn_new_dists_i = knn_new_dists[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()
                    # knn_ids_i = knn_ids[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()
                    val_ids_i = val_ids[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()
                    val_count_i = val_counts[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()

                if 'knn_dstore' in kwargs:
                    knn_probs_i = yhat_knn_prob[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
                else:
                    knn_probs_i = None
            else:
                lm_probs_i = knn_probs_i = None
                ######################################
                if self.args.analyze_knn:
                    lm_probs_i = lm_probs[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu() # We always have lm_probs
                    if self.args.knnlm:
                        knn_old_dists_i = knn_old_dists[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()
                        # knn_new_dists_i = knn_new_dists[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()
                        # knn_ids_i = knn_ids[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()
                        val_ids_i = val_ids[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()      
                        knn_probs_i = yhat_knn_prob[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
                        val_count_i = val_counts[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()

            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            nwp_answer_i = 0
            nwp_correct_i = []
            vars_i = []
            if self.args.count_acc:
                if nwp_correct != []:
                    nwp_correct_i = nwp_correct[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
                    nwp_answer_i = torch.sum(nwp_correct_i).item()
                    if self.args.deviation:
                        vars_i = vars[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
            if lm_vocab_probs is not None:
                lm_vocab_probs_i = lm_vocab_probs[i][start_idxs[i] + tgt_len - 1].cpu()
                knn_vocab_probs_i = knn_vocab_probs[i][start_idxs[i] + tgt_len - 1].cpu()
                net_lambda_i = net_lambda[i][start_idxs[i] + tgt_len - 1].cpu()
                # print(net_lambda_i)
            else:
                lm_vocab_probs_i, knn_vocab_probs_i, net_lambda_i = None, None, None
            # print(nwp_answer)
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
                'knn_scores': knn_probs_i,
                'lm_scores': lm_probs_i,
                'lm_entropy': lm_entropy_i,
                'lm_max': lm_max_i,
                'lm_context': lm_context_i,
                'knn_old_dists': knn_old_dists_i,
                'knn_new_dists': knn_new_dists_i,
                'knn_ids': knn_ids_i,
                'val_ids': val_ids_i,
                'val_count': val_count_i,
                'dstore_keys': decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:] if self.args.save_knnlm_dstore else None,
                'nwp_answer': nwp_answer_i,
                'nwp_correct': nwp_correct_i,
                'deviation': vars_i,
                'lm_vocab_probs': lm_vocab_probs_i,
                'knn_vocab_probs': knn_vocab_probs_i,
                'net_lambda': net_lambda_i,
            }])

        # print(f'processing output consumes {time.time() - cur_time} seconds')
        return hypos
