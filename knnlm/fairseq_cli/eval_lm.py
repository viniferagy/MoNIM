#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os
import json
import time
import pickle
import torch
import numpy as np
from numpy.lib.format import open_memmap

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.knnlm import KNN_Dstore
# from prompt import EvaluatingWrapper, load_test_data

def read_jsonl(in_file):
    questions = []
    with open(in_file) as fin:
        for line in fin:
            question = json.loads(line)
            questions.append(question)
    return questions

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('fairseq_cli.eval_lm')


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """ increments counters for the sum of log probs of current word and next
            word (given context ending at current word). Since the next word might be at the end of the example,
            or it might be not counted because it is not an ending subword unit,
            also keeps track of how many of those we have seen """
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(self.word, self.count, self.log_prob, self.is_bpe,
                                               self.next_word_prob, self.count - self.missing_next_words)


def main(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'

    utils.import_user_module(parsed_args)

    if parsed_args.save_knnlm_dstore:
        logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(parsed_args.path))
    # args.tokens_per_sample: 1024 -> args.max_tokens
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
    )
    # import pdb; pdb.set_trace()

    logger.info(f'training max tokens {args.max_tokens}, tokens per sample {args.tokens_per_sample}, break mode {args.sample_break_mode}')

    for arg in vars(parsed_args).keys():
        if arg not in {
            'self_target', 'future_target', 'past_target', 'tokens_per_sample',
            'output_size_dictionary', 'add_bos_token',
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    args.tokens_per_sample -= args.context_window
    task = tasks.setup_task(args)

    args.ar_feat_type = args.ar_feat_type.split(',')

    if (args.ar_ckpt != 'none') and args.ar_freq_dict != '':
        if 'freq' in args.ar_feat_type:
            print('loading freq cache')
            freq_id_cache = pickle.load(open(os.path.join(args.ar_freq_dict, 'freq_cache_id.pickle'), 'rb'))
        else:
            freq_id_cache = None

        if 'fert' in args.ar_feat_type:
            print('loading fert cache')
            fertility_id_cache = pickle.load(open(os.path.join(args.ar_freq_dict, 'fertility_cache_id.pickle'), 'rb'))
        else:
            fertility_id_cache = None
        # fertility_id_cache=None
        # fertility_id_cache = freq_id_cache
    else:
        freq_id_cache = fertility_id_cache = None

    if args.context_window > 0:
        # Load dataset splits
        task.load_dataset(args.gen_subset)
        dataset = task.dataset(args.gen_subset)

        dataset = LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=args.tokens_per_sample,
            context_window=args.context_window,
            pad_idx=task.source_dictionary.pad(),
            freq=freq_id_cache,
            fert=fertility_id_cache,
            knnlm_feat_csize=args.knnlm_feat_csize,
        )

    else:
        task.load_dataset(args.gen_subset,
                          freq=freq_id_cache,
                          fert=fertility_id_cache,
                          knnlm_feat_csize=args.knnlm_feat_csize,
                          )

        dataset = task.dataset(args.gen_subset)

    logger.info('{} {} {} examples'.format(args.data, args.gen_subset, len(dataset)))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    logger.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # ---- Melongena stop words erasing code ----
    def stopword(size):
        if size == 'sm':
            file = open(f"{args.output_dir}/stop-sm.txt", "r")
            content = file.read()
            stops = set(content.split())
        elif size == 'mid':
            file = open(f"{args.output_dir}/stop-mid.txt", "r")
            content = file.read()
            stops = set(content.split())
        elif size == 'lg':
            import requests
            content = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
            stops = set(content.decode().splitlines()) 
        elif size.split('_')[0] == 'idf':
            _, idf_name, idf_threshold = size.split('_') 
            import bisect
            idf_threshold = float(idf_threshold) # 1.5: 33, 2.0: 101 words
            print(f'{idf_name} idf threshold is {idf_threshold}')
            d = json.load(open(f"{args.output_dir}/idf_{idf_name}.txt"))
            stops = list(d.keys())[:bisect.bisect(list(d.values()), idf_threshold)]
            
        return stops
    
    if args.prune_name:
        prune = True
        prune_name = args.prune_name
        preserved_rate = 1 - args.prune_rate
    elif args.stop_size != "":
        prune = True
        prune_name = 'stop'
        print(f'Use stop size {args.stop_size}')
        stop = ' '.join(stopword(args.stop_size))
        stop_ids = task.target_dictionary.encode_line(stop, add_if_not_exist=False)
        # print(stop_ids)
        # stop = task.target_dictionary.string(stop_ids)
        # print(stop)
        # raise AssertionError
        stop_ids = utils.move_to_cuda(stop_ids)
        args.stop = stop_ids
    # ---- Melongena Pruning Code ----
    elif args.prune_threshold != 0 or args.prune_rate > 1e-5:
        prune = True
        preserved_rate = 1 - args.prune_rate
        if args.prune_threshold != 0:
            prune_name = 'threshold'
        else:
            prune_name = 'random'
    else:
        prune = False
        prune_name = None
    if prune:
        print(f'----- {prune_name} prune -----')
    # ---- Melongena Pruning Code ----

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch, args=args)

    score_sum = 0.
    count = 0
    nwp_answers = 0

    if args.remove_bpe is not None:
        if args.remove_bpe == 'sentencepiece':
            raise NotImplementedError
        else:
            bpe_cont = args.remove_bpe.rstrip()
            bpe_toks = {
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            }
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    if args.knnlm and args.save_knnlm_dstore:
        if not args.continual:
            raise ValueError("Cannot use knnlm while trying to build the datastore!")

    if args.knnlm:
        knn_dstore = KNN_Dstore(args, task.target_dictionary)
        logger.info('Reading knn datastore from: {}, index: {}'.format(args.infer_dstore_path, args.index_file))

    if args.save_feature is not None:
        fout_ctxt = open(f'{args.save_feature}_ctxt.{args.shard_id}.jsonl', 'w')
        fout_extra = open(f'{args.save_feature}_others.{args.shard_id}.jsonl', 'w')

        ngram = args.knnlm_feat_csize
        prev = [task.target_dictionary.index('</s>')] * ngram

        freq_cache = os.path.join(args.ar_feat_cache, 'freq_cache_id.pickle')
        fertility_cache = os.path.join(args.ar_feat_cache, 'fertility_cache_id.pickle')

        if os.path.isfile(freq_cache):
            print('loading freq cnt from cache')
            with open(freq_cache, 'rb') as pf:
                freq_cnt = pickle.load(pf)
        else:
            raise ValueError('frequency cache file not existing')

        if os.path.isfile(fertility_cache):
            print('loading fertility cnt from cache')
            with open(fertility_cache, 'rb') as pf:
                fertility_cnt = pickle.load(pf)
        else:
            raise ValueError('fertility cache file not existing')
    if args.infer_dstore_path:
        debug_file = os.path.basename(args.infer_dstore_path)
        if args.knnlm:
            debug_file = os.path.basename(args.infer_dstore_path).replace('dstore_', '')
    total_prob = []
    total_knn = []
    total_lm_prob = []
    total_knn_prob = []
    total_token = []
    total_nwp_correct = []
    total_vars = []
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()

        if args.save_knnlm_dstore:
            print('keytype being saved: ', args.knn_keytype)
            fp = 16 if args.dstore_fp16 else 32
            print(f'Saving fp{fp} in {args.save_dstore_path}')
            dstore_keys = np.zeros(shape=(math.ceil(args.dstore_size/args.num_shards), args.decoder_embed_dim), dtype=eval(f'np.float{fp}'))
            dstore_vals = np.zeros(shape=(math.ceil(args.dstore_size/args.num_shards), 1), dtype=np.int64)
            # dstore_keys = zarr.open(f'{args.save_dstore_path}_keys.today.npy.{args.shard_id}', dtype=eval(f'np.float{fp}'), mode='w', shape=(args.dstore_size, args.decoder_embed_dim))
            # dstore_vals = zarr.open(f'{args.save_dstore_path}_vals.today.npy.{args.shard_id}', dtype=np.int64, mode='w', shape=(args.dstore_size, 1))
            if prune_name == 'threshold' and args.prune_rate < 1.0:
                print('need weight')
                # dstore_weights = zarr.open(f'{args.save_dstore_path}_weights.today.npy.{args.shard_id}', dtype=eval(f'np.float{fp}'), mode='w+', shape=(args.dstore_size, 1))
            else:
                dstore_weights = None
            

        dstore_idx = 0
        dstore_scanned = 0
        useful_counts = 0
        if args.qa:
            total_qa_ppl = [0] * len(dataset)
            line_index = 0
            pos_index = 0
            # print(ppl_perline)
        if args.end_task:
            examples, closed_label_space = load_test_data(args)
            args.scoring = 'log_softmax'
            eval_wrapper = EvaluatingWrapper(dictionary=task.target_dictionary, examples=examples, args=args)
            lm_vocab_probs, knn_vocab_probs = [0] * len(dataset), [0] * len(dataset)
            net_lambda = [0] * len(dataset)
        try:
            for ex_i, sample in enumerate(t):
                # Early stop, for debugging
                if args.save_knnlm_dstore and dstore_scanned >= args.dstore_size:
                    print('Early stopped')
                    break
                if 'net_input' not in sample:
                    continue
                sample = utils.move_to_cuda(sample) if use_cuda else sample

                gen_timer.start()
                # print('scorer generate')
                if args.knnlm:
                    hypos = scorer.generate(models, sample, knn_dstore=knn_dstore)
                else:
                    hypos = scorer.generate(models, sample)
                gen_timer.stop(sample['ntokens'])

                for i, hypos_i in enumerate(hypos):
                    hypo = hypos_i[0]

                    sample_id = sample['id'][i]

                    tokens = hypo['tokens']
                    tgt_len = tokens.numel()
                    pos_scores = hypo['positional_scores'].float()

                    if args.save_knnlm_dstore:
                        # if use eos, we should change some constraints
                        if hypo['dstore_keys'].shape[0] == args.tokens_per_sample or args.sample_break_mode == 'eos':
                            # Last case
                            if dstore_idx + hypo['dstore_keys'].shape[0] > args.dstore_size:
                                last_size = args.dstore_size - dstore_idx
                                hypo['dstore_keys'] = hypo['dstore_keys'][:last_size]
                                hypo['tokens'] = hypo['tokens'][:last_size]
                                pos_scores = hypo['positional_scores'][:last_size].float()
                            if prune:
                                if prune_name == "improve": # 试着多prune一些 knn效果不好，说明dstore内这附近的sample少，保留
                                    if args.continual:
                                        preserved_index = hypo['knn_scores'] <= hypo['lm_scores']
                                    else:
                                        preserved_index = torch.full_like(hypo['tokens'], fill_value=True)
                                    # print(preserved_index.shape, preserved_index.sum())
                                    cur_dstore_weights = torch.ones(size=(pos_scores.shape[0], 1))
                                    cur_dstore_weights = cur_dstore_weights[preserved_index.cpu()]
                                elif prune_name == 'random':
                                    preserved_index = np.random.choice(range(pos_scores.shape[0]), 
                                                        size=int(preserved_rate*pos_scores.shape[0]),
                                                        replace=False)
                                elif prune_name == 'threshold':
                                    # Px1
                                    # if args.continual and args.analyze_knn:
                                    #     unimproved_part = (hypo['knn_scores'] <= hypo['lm_scores']).cuda()
                                    #     useful_part = (hypo['lm_scores'] < args.prune_threshold).cuda()
                                    # else:
                                    #     unimproved_part = torch.full_like(pos_scores, dtype=torch.bool, fill_value=True).cuda()
                                    #     useful_part = (pos_scores < args.prune_threshold).cuda()
                                    if args.prune_threshold < 0:
                                        useful_part = (pos_scores < args.prune_threshold).cuda()
                                    else:
                                        useful_part = (pos_scores > -args.prune_threshold).cuda()
                                    # prune_index = torch.logical_and(~useful_part, unimproved_part).nonzero()
                                    prune_index = (~useful_part).nonzero()
                                    # import pdb; pdb.set_trace()
                                    np.random.seed(args.seed)
                                    if preserved_rate > 1e-5:
                                        sample_index = np.random.choice(range(prune_index.shape[0]), 
                                                        size=int(preserved_rate*prune_index.shape[0]), 
                                                        replace=False)
                                    else:
                                        sample_index = []
                                    # P'x1 -> P'
                                    preserved_index = torch.concat([
                                        # torch.logical_and(useful_part, unimproved_part).nonzero(),
                                        useful_part.nonzero(), 
                                        prune_index[sample_index]
                                        ], dim=0).squeeze(-1)
                                    if dstore_weights is not None:
                                        cur_dstore_weights = torch.ones(size=(pos_scores.shape[0], 1))
                                        cur_dstore_weights[prune_index[sample_index]] = 1 / preserved_rate
                                        cur_dstore_weights = cur_dstore_weights[preserved_index.cpu()]
                                elif prune_name == 'stop':
                                    preserved_mask = torch.isin(hypo['tokens'], stop_ids, invert=True)
                                    preserved_index = torch.nonzero(preserved_mask).squeeze(-1)
                                
                                key_pruned = hypo['dstore_keys'][preserved_index]
                                val_pruned = hypo['tokens'][preserved_index]
                            else:
                                useful_count = (pos_scores < -1.5).sum().item()
                                useful_counts += useful_count
                                # print(f'Useful count for -1.5in0,0: {useful_count}')
                                # a = json.load(open(f'{args.output_dir}/size.json', 'r'))
                                # *_, dataset, _, dstore_prefix = args.save_dstore_path.split('/')
                                # info_list = dstore_prefix.split('_')
                                # ckpt_name, date, prune_param = info_list[0], info_list[1], info_list[-1]
                                # a[dataset][ckpt_name][date]['-1.5in0,0'] = useful_count
                                # json.dump(a, open(f'{args.output_dir}/size.json', 'w+'), indent=4)

                                val_pruned = hypo['tokens']
                                key_pruned = hypo['dstore_keys'][:val_pruned.shape[0]]
                            shape = val_pruned.shape
                            # import pdb; pdb.set_trace()
                            dstore_keys[dstore_idx:shape[0]+dstore_idx] = key_pruned.view(-1, args.decoder_embed_dim).cpu().numpy().astype(np.float16 if args.dstore_fp16 else np.float32)
                            dstore_vals[dstore_idx:shape[0]+dstore_idx] = val_pruned.view(
                                -1, 1).cpu().numpy().astype(int)
                            if dstore_weights is not None:
                                # print(cur_dstore_weights.shape)
                                dstore_weights[dstore_idx:shape[0]+dstore_idx] = cur_dstore_weights

                            dstore_idx += shape[0]
                            dstore_scanned += hypo['tokens'].shape[0]
                        else:
                            print('Skipping this one with shape', hypo['dstore_keys'].shape[0])
                    else: # During inference, simply count the amount below threshold
                        dstore_idx += (pos_scores < args.prune_threshold).sum().item()
                    if args.add_bos_token:
                        assert hypo['tokens'][0].item() == task.target_dictionary.bos()
                        tokens = tokens[1:]
                        pos_scores = pos_scores[1:]

                    if args.save_feature is not None:
                        for k in range(len(hypo['tokens'])):
                            # tok = task.target_dictionary[hypo['tokens'][k].item()]
                            tok = hypo['tokens'][k].item()
                            prev = prev[-ngram:]
                            hypo_others_tmp = {'tgt': tok,
                                        'tgt_token': task.target_dictionary[tok],
                                        'int_s': hypo['positional_scores'][k].item(),
                                        # 'knn_s': hypo['knn_scores'][k].item() if hypo['knn_scores'] is not None else None,
                                        'lm_s': hypo['lm_scores'][k].item(),
                                        'lm_ent': hypo['lm_entropy'][k].item(),
                                        'lm_max': np.exp(hypo['lm_max'][k].item()),
                                        'freq': [freq_cnt[tuple(prev[-j:])] for j in range(1, ngram + 1)],
                                        'fert': [fertility_cnt[tuple(prev[-j:])] for j in range(1, ngram + 1)],
                                        # 'knn_dists': hypo['knn_dists'][k].tolist(),
                                }
                            if args.analyze_knn:
                                hypo_others_tmp.update({
                                    'old_d': hypo['knn_old_dists'][k].tolist(),
                                    'val_id': hypo['val_ids'][k].tolist(),
                                    'val_count': hypo['val_count'][k].tolist(),
                                    # 'knn': task.target_dictionary.string(hypo['knn_ids'][k]),
                                    })
                            prev.append(tok)

                            hypo_ctxt_tmp = {'ctxt': hypo['lm_context'][k].tolist()}
                            fout_ctxt.write(json.dumps(hypo_ctxt_tmp, ensure_ascii=False))
                            fout_ctxt.write('\n')
                            fout_ctxt.flush()

                            fout_extra.write(json.dumps(hypo_others_tmp, ensure_ascii=False))
                            fout_extra.write('\n')
                            fout_extra.flush()

                    skipped_toks = 0
                    if bpe_toks is not None:
                        for i in range(tgt_len - 1):
                            if tokens[i].item() in bpe_toks:
                                skipped_toks += 1
                                pos_scores[i + 1] += pos_scores[i]
                                pos_scores[i] = 0

                    if args.analyze_knn:
                        if args.stop_size and prune_name == 'stop':
                            pruned_mask = torch.isin(hypo['tokens'], stop_ids)
                            pruned_index = torch.nonzero(pruned_mask).squeeze(-1)
                        # total_knn.append(hypo['knn_old_dists'][:, :3].tolist())
                            total_knn_prob += hypo['knn_scores'][pruned_index].tolist()
                            total_lm_prob += hypo['lm_scores'][pruned_index].tolist()
                            total_token += hypo['tokens'][pruned_index].tolist()
                        # assert hypo['knn_old_dists'].shape[0] == hypo['tokens'].shape[0]
                        else:
                            if args.knnlm:
                                total_knn_prob += hypo['knn_scores'].tolist()
                            total_lm_prob += hypo['lm_scores'].tolist()
                            total_token += hypo['tokens'].tolist()
                    if args.count_acc:
                        total_nwp_correct += hypo['nwp_correct'].tolist()
                        if args.deviation:
                            total_vars += hypo['deviation'].tolist()
                        
                    total_prob += pos_scores.tolist()
                    score_sum += pos_scores.sum().cpu()
                    count += pos_scores.numel() - skipped_toks

                    if args.qa:
                        score_perline = pos_scores.sum().cpu()
                        count_perline = pos_scores.numel() - skipped_toks
                        sample_index = sample['id'][i].item()
                        if count_perline == 1: # '/n'
                            total_qa_ppl[sample_index] = -1
                        else: 
                            loss_perline = -score_perline / count_perline / math.log(2)  # convert to base 2
                            loss_perline = loss_perline.item()
                            total_qa_ppl[sample_index] = 2**loss_perline
                    if args.end_task:
                        lm_vocab_probs[sample['id'][i]] = hypo['lm_vocab_probs']
                        knn_vocab_probs[sample['id'][i]] = hypo['knn_vocab_probs']
                        net_lambda[sample['id'][i]] = hypo['net_lambda']
                    nwp_answers += hypo['nwp_answer']

                    if args.output_word_probs or args.output_word_stats:
                        w = ''
                        word_prob = []
                        is_bpe = False
                        for i in range(len(tokens)):
                            w_ind = tokens[i].item()
                            w += task.source_dictionary[w_ind]
                            if bpe_toks is not None and w_ind in bpe_toks:
                                w = w[:-bpe_len]
                                is_bpe = True
                            else:
                                word_prob.append((w, pos_scores[i].item()))

                                next_prob = None
                                ind = i + 1
                                while ind < len(tokens):
                                    if pos_scores[ind].item() != 0:
                                        next_prob = pos_scores[ind]
                                        break
                                    ind += 1

                                word_stats.setdefault(w, WordStat(w, is_bpe)).add(pos_scores[i].item(), next_prob)
                                is_bpe = False
                                w = ''
                        if args.output_word_probs:
                            logger.info(
                                str(int(sample_id)) + " "
                                + ('\t'.join('{} [{:2f}]'.format(x[0], x[1]) for x in word_prob))
                            )

                wps_meter.update(sample['ntokens'])
                t.log({'wps': round(wps_meter.avg)})   
        except Exception as e:
            print(e.args)
            print(str(e))
            print(repr(e))
            pass
        if args.qa:
            if args.qa == 'CommonQA':
                choice_num = 5
            else:
                choice_num = 4
            qa_ppl_perline = np.full((len(dataset), choice_num), fill_value=10000.0, dtype=np.float64)
            # print(total_qa_ppl)
            for i in range(len(total_qa_ppl)):
                if total_qa_ppl[i] == -1:
                    line_index += 1
                    pos_index = 0
                else:
                    qa_ppl_perline[line_index][pos_index] = total_qa_ppl[i]
                    pos_index += 1
            best_perquestion = np.argmin(qa_ppl_perline, axis=-1)[:line_index+1]
            print(best_perquestion)
            if args.qa == 'CommonQA':
                gold_a = np.genfromtxt(f'{args.data}/../answer.txt')
                print(gold_a)
            else:
                questions = read_jsonl(f'{args.data}/../../../../realtime/backnumber/total/{args.qa}_qa.jsonl')
                gold_a = [int(q['answer'][0]) for q in questions]
            from sklearn.metrics import classification_report, accuracy_score
            print(classification_report(gold_a, best_perquestion))
            print(f'Acc: {accuracy_score(gold_a, best_perquestion):.4f}')
            path = f'{args.output_dir}/debug/qa'
            if not os.path.exists(path):
                os.makedirs(path)
            file = f'{path}/{debug_file}.txt'
            with open(f'{file}', 'a+') as f:
                f.write(f'lmbda: {args.lmbda}, temp: {args.knn_temp}, acc: {accuracy_score(gold_a, best_perquestion)}\n')
        if args.end_task:
            print(len(dataset))
            domain_lm_probs = lm_vocab_probs[0]
            domain_knn_probs = knn_vocab_probs[1]
            lm_vocab_probs = lm_vocab_probs[0::3][1:]
            knn_vocab_probs = knn_vocab_probs[1::3][1:]
            print(len(lm_vocab_probs))
            eval_wrapper.score(lm_vocab_probs, knn_vocab_probs, domain_lm_probs, domain_knn_probs, net_lambda)
    
    # save size info to size.json
    if args.save_knnlm_dstore:
        # *_, dataset, _, dstore_prefix = args.save_dstore_path.split('/')
        # info_list = dstore_prefix.split('_')
        # ckpt_name, date, prune_param = info_list[0], info_list[1], info_list[-1]
        # try:
        #     a = json.load(open(f'{args.output_dir}/size.json', 'r'))
        # except:
        #     a = {}
        # if not a.get(dataset):
        #     a[dataset] = dict()
        # if not a[dataset].get(ckpt_name):
        #     a[dataset][ckpt_name] = dict()
        # if not a[dataset][ckpt_name].get(date):
        #     a[dataset][ckpt_name][date] = dict()
        # a[dataset][ckpt_name][date][f"{prune_param}-now"] = dstore_idx
        # json.dump(a, open(f'{args.output_dir}/size.json', 'w+'), indent=4)
        print("Keys", dstore_idx, dstore_keys.dtype)
        print("Vals", dstore_idx, dstore_vals.dtype)
        # np.save(f'{args.save_dstore_path}_keys.today.{args.shard_id}.npy', dstore_keys)
        # np.save(f'{args.save_dstore_path}_vals.today.{args.shard_id}.npy', dstore_vals)
        mm_dstore_keys = open_memmap(f'{args.save_dstore_path}_keys.today.{args.shard_id}.npy', dtype=eval(f'np.float{fp}'), mode='w+', shape=(dstore_idx, args.decoder_embed_dim))
        mm_dstore_vals = open_memmap(f'{args.save_dstore_path}_vals.today.{args.shard_id}.npy', dtype=np.int64, mode='w+', shape=(dstore_idx, 1))
        mm_dstore_keys[:] = dstore_keys[:dstore_idx]
        mm_dstore_vals[:] = dstore_vals[:dstore_idx]
        mm_dstore_keys.flush()
        mm_dstore_vals.flush()
        # dstore_keys.resize((dstore_idx, args.decoder_embed_dim))
        # dstore_vals.resize((dstore_idx, 1))

    if count == 0 and args.save_feature is not None:
        logger.info(
            'No tokens found for feature shard %s; wrote empty feature files and exiting',
            args.shard_id,
        )
        fout_ctxt.close()
        fout_extra.close()
        return

    avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
    nwp_acc = nwp_answers / count

    logger.info(f'Save {dstore_idx}/{count} ({dstore_idx/count:.4f}) tokens; Useful counts: {useful_counts}')
    logger.info('Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(
        gen_timer.n, gen_timer.sum, 1. / gen_timer.avg
    ))
    logger.info('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
        avg_nll_loss, 2**avg_nll_loss
    ))
    logger.info('next word prediction accuracy: {:.3f}'.format(
        nwp_acc
    ))
    open(f'useful{args.shard_id}.tmp', 'w+').write(str(useful_counts))

    if args.output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            logger.info(ws)
    if args.output_dir and not args.save_feature:
        if args.knnlm and not args.save_knnlm_dstore:
            path_ppl = f'{args.output_dir}/debug/ppl'
            path_pwd = f'{args.output_dir}/debug/pwd'
            if not os.path.exists(path_ppl):
                os.makedirs(path_ppl)
            if not os.path.exists(path_pwd):
                os.makedirs(path_pwd)
            file_ppl = f'{path_ppl}/{debug_file}.txt'
            file_pwd = f'{path_pwd}/{debug_file}.txt'
            with open(f'{file_ppl}', 'a+') as f:
                f.write(f'lmbda: {args.lmbda}, temp: {args.knn_temp} Loss (base 2): {avg_nll_loss:.4f}, Perplexity: {2**avg_nll_loss:.2f}\n')
            with open(f'{file_pwd}', 'a+') as f:
                f.write(f'lmbda: {args.lmbda}, temp: {args.knn_temp} next word prediction accuracy: {nwp_acc:.3f}\n')

        if args.count_acc:
            if not os.path.exists(f'{args.output_dir}/debug/count_acc'):
                os.makedirs(f'{args.output_dir}/debug/count_acc')
            with open(f'{args.output_dir}/debug/count_acc/{debug_file}.txt', 'w+') as f:
                json.dump(total_nwp_correct, f)
            if args.deviation:  
                if not os.path.exists(f'{args.output_dir}/debug/vars'):
                    os.makedirs(f'{args.output_dir}/debug/vars')
                with open(f'{args.output_dir}/debug/vars/{debug_file}.txt', 'w+') as f:
                    json.dump(total_vars, f)

        if args.analyze_knn:
            print('Saving debug information...')
            exit()
            # if args.stop_size:
            #     print(f'We have {len(total_knn_prob)} stop tokens in {args.gen_subset} set!')
            # if args.knnlm:
            #     if not os.path.exists(f'{args.output_dir}/debug/knn_prob'):
            #         os.makedirs(f'{args.output_dir}/debug/knn_prob')
            #     with open(f'{args.output_dir}/debug/knn_prob/{debug_file}.txt', 'w+') as f:
            #         json.dump(total_knn_prob, f)
            # if not os.path.exists(f'{args.output_dir}/debug/lm_prob'):
            #     os.makedirs(f'{args.output_dir}/debug/lm_prob')
            # with open(f'{args.output_dir}/debug/lm_prob/{debug_file}.txt', 'w+') as f:
            #     json.dump(total_lm_prob, f)
            # if not os.path.exists(f'{args.output_dir}/debug/token'):
            #     os.makedirs(f'{args.output_dir}/debug/token')
            # with open(f'{args.output_dir}/debug/token/{args.date}_{args.gen_subset}.txt', 'w+') as f:
            #     json.dump(total_token, f)    


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    if args.save_knnlm_dstore or args.knnlm:
        if args.dstore_size == 0: # read dstore size from json file
            # import json
            a = json.load(open(f'{args.output_dir}/size.json', 'r'))
            if args.save_knnlm_dstore:
                *_, dataset, _, dstore_prefix = args.save_dstore_path.split('/')
                info_list = dstore_prefix.split('_')
                print(info_list)
                ckpt_name, date, prune_param = info_list[0], info_list[1], info_list[-1]
                args.dstore_size = int(a[dataset][ckpt_name][date]['total'])
            else:
                *_, dataset, _, dstore_prefix = args.infer_dstore_path.split('/')
                info_list = dstore_prefix.split('_')
                print(info_list)
                ckpt_name, date, prune_param = info_list[0], info_list[1], info_list[-1]
                args.dstore_size = int(a[dataset][ckpt_name][date][prune_param])
            # print(info_list)
            print('Dstore size from json:', args.dstore_size)
    main(args)


if __name__ == '__main__':
    cli_main()
