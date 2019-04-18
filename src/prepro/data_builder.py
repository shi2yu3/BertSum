import gc
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from os.path import join as pjoin

import torch
from multiprocess import Pool
from pytorch_pretrained_bert import BertTokenizer

from others.logging import logger
from others.utils import clean
from prepro.utils import _get_word_ngrams


END_TOKENS = ['.', '!', '?'] # acceptable ways to end a sentence

def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    data = json.load(open(p, encoding='utf-8'))
    for sent in data['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
        if (flag):
            tgt.append(tokens)
            flag = False
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return data['docId'], source, tgt


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = [0.0] * len(abstract_sent_list)
    max_idx = [[]] * len(abstract_sent_list)
    abstract = [_rouge_clean(' '.join(a)).split() for a in abstract_sent_list]
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = [_get_word_ngrams(1, [a]) for a in abstract]
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = [_get_word_ngrams(2, [a]) for a in abstract]

    impossible_sents = []
    for s in range(summary_size):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            best_match_idx = -1
            best_match_rouge = 0.0
            for i in range(len(reference_1grams)):
                rouge_1 = cal_rouge(candidates_1, reference_1grams[i])['f']
                rouge_2 = cal_rouge(candidates_2, reference_2grams[i])['f']
                rouge_score = (rouge_1 + rouge_2) / 2.0
                if rouge_score > best_match_rouge:
                    best_match_idx = i
                    best_match_rouge = rouge_score
            if (s == 0 and best_match_idx == -1):
                impossible_sents.append(c[0])
            if best_match_idx >= 0 and best_match_rouge > max_rouge[best_match_idx]:
                max_idx[best_match_idx] = list(c)
                max_rouge[best_match_idx] = best_match_rouge

    return max_idx, max_rouge


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess(self, src, tgt, oracle_ids):

        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        labels = [0] * len(src)
        for l in oracle_ids:
            labels[l] = 1

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        if (len(src) < self.args.min_nsents):
            return None
        if (len(labels) == 0):
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = os.path.basename(json_f)
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        logger.info(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)
    os.makedirs(tokenized_stories_dir, exist_ok=True)

    logger.info("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    logger.info("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP' ,'-annotators', 'tokenize,ssplit', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat', 'json', '-outputDirectory', tokenized_stories_dir]
    logger.info("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    logger.info("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    logger.info("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def _format_to_bert(params):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        if (args.oracle_mode == 'greedy'):
            oracle_ids = greedy_selection(source, tgt, 3)
        elif (args.oracle_mode == 'combination'):
            oracle_ids = combination_selection(source, tgt, 3)
        b_data = bert.preprocess(source, tgt, oracle_ids)
        if (b_data is None):
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                json.dump(dataset, open(pt_file, 'w'), indent=2)
                p_ct += 1
                dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            json.dump(dataset, open(pt_file, 'w'), indent=2)
            p_ct += 1
            dataset = []


def _format_to_lines(params):
    f, args = params
    logger.info(f)
    docId, source, tgt = load_json(f, args.lower)
    return {'docId': docId, 'src': source, 'tgt': tgt}


def fix_missing_period(args):
    input_dir = os.path.abspath(args.raw_path)
    output_dir = os.path.abspath(args.save_path)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Fixing missing period in %s and saving in %s..." % (input_dir, output_dir))
    stories = os.listdir(input_dir)
    for s in stories:
        if (not s.endswith('story')):
            continue
        _fix_missing_period(os.path.join(input_dir, s), os.path.join(output_dir, s))

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_inputs = len(os.listdir(input_dir))
    num_outputs = len(os.listdir(output_dir))
    if num_inputs != num_outputs:
        raise Exception(
            "The output directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during processing?" % (
            output_dir, num_outputs, input_dir, num_inputs))
    logger.info("Successfully finished fixing missing period %s to %s.\n" % (input_dir, output_dir))


def _fix_missing_period(s, t):
    """Adds a period to a line that is missing a period"""
    logger.info(s)
    lines = []
    with open(s, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and "@highlight" not in line.lower() and line[-1] not in END_TOKENS:
                line += " ."
            lines.append(line)

    with open(t, "w", encoding='utf-8') as f:
        f.write("\n".join(lines))


def analysis(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = os.path.basename(json_f)
            a_lst.append((json_f, args, pjoin(args.save_path, real_name)))
        logger.info(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_analysis, a_lst):
            pass

        pool.close()
        pool.join()


def _analysis(params):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    stats = {
        'document': {'#documents': 0, '#documents with overlapped highlights': 0},
        'highlight': {'rouge all': {}}}
    for i in range(11):
        r = f"rouge {i / 10:.1f}"
        stats['highlight'][r] = {}
        stats['highlight'][r]['#highlights'] = 0
        stats['highlight'][r]['#single-segment highlights'] = 0
        stats['highlight'][r]['#multi-segment highlights'] = 0
        for j in range(4):
            stats['highlight'][r][f"#{j}-sentence highlights"] = 0

    dataset = []
    for num, d in enumerate(jobs):
        source, tgt = d['src'], d['tgt']
        if args.lower:
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        ids, rouges = greedy_combination_selection(source, tgt, 3)
        dataset.append({'docId': d['docId'], 'segIds': ids, 'rouge': rouges})
        overlap = False
        for i in range(len(ids)):
            r = f"rouge {int(rouges[i]*10)/10:.1f}"
            n = f"#{len(ids[i])}-sentence highlights"
            stats['highlight'][r][n] += 1
            if ids[i]:
                if len(ids[i]) == max(ids[i]) - min(ids[i]) + 1:
                    stats['highlight'][r]['#single-segment highlights'] += 1
                else:
                    stats['highlight'][r]['#multi-segment highlights'] += 1
            if not overlap:
                for j in range(i + 1, len(ids)):
                    if len(set(ids[i]).intersection(ids[j])) > 0:
                        overlap = True
            stats['highlight'][r]['#highlights'] += 1
        if overlap:
            stats['document']['#documents with overlapped highlights'] += 1
        stats['document']['#documents'] += 1

        if (num + 1) % 1000 == 0:
            highlight_stats = stats['highlight']
            highlight_stats['rouge all']['#highlights'] = 0
            highlight_stats['rouge all']['#single-segment highlights'] = 0
            highlight_stats['rouge all']['#multi-segment highlights'] = 0
            for i in range(4):
                highlight_stats['rouge all'][f"#{i}-sentence highlights"] = 0
            for i in range(11):
                r = f"rouge {i / 10:.1f}"
                highlight_stats['rouge all']['#highlights'] += highlight_stats[r]['#highlights']
                highlight_stats['rouge all']['#single-segment highlights'] += highlight_stats[r][
                    '#single-segment highlights']
                highlight_stats['rouge all']['#multi-segment highlights'] += highlight_stats[r][
                    '#multi-segment highlights']
                for i in range(4):
                    highlight_stats['rouge all'][f"#{i}-sentence highlights"] += highlight_stats[r][
                        f"#{i}-sentence highlights"]
            logger.info(f'Saving {num+1} results to {save_file}')
            json.dump([stats] + dataset, open(save_file, "w"), indent=2)

    highlight_stats = stats['highlight']
    highlight_stats['rouge all']['#highlights'] = 0
    highlight_stats['rouge all']['#single-segment highlights'] = 0
    highlight_stats['rouge all']['#multi-segment highlights'] = 0
    for i in range(4):
        highlight_stats['rouge all'][f"#{i}-sentence highlights"] = 0
    for i in range(11):
        r = f"rouge {i / 10:.1f}"
        highlight_stats['rouge all']['#highlights'] += highlight_stats[r]['#highlights']
        highlight_stats['rouge all']['#single-segment highlights'] += highlight_stats[r]['#single-segment highlights']
        highlight_stats['rouge all']['#multi-segment highlights'] += highlight_stats[r]['#multi-segment highlights']
        for i in range(4):
            highlight_stats['rouge all'][f"#{i}-sentence highlights"] += highlight_stats[r][f"#{i}-sentence highlights"]

    logger.info('Saving to %s' % save_file)
    json.dump([stats] + dataset, open(save_file, "w"), indent=2)


def format_to_translation(args):
    os.makedirs(args.save_path, exist_ok=True)
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        for d in pool.imap_unordered(_format_to_translation, a_lst):
            dataset.append(d)
        pool.close()
        pool.join()

        src_file = os.path.join(args.save_path, f"{corpus_type}.src.txt")
        tgt_file = os.path.join(args.save_path, f"{corpus_type}.tgt.txt")
        with open(src_file, "w", encoding='utf-8') as src:
            with open(tgt_file, "w", encoding='utf-8') as tgt:
                for i in range(len(dataset)):
                    src.write(f"{dataset[i]['src']}\n")
                    tgt.write(f"{dataset[i]['tgt']}\n")


def _format_to_translation(params):
    f, args = params
    logger.info(f)
    docId, source, tgt = load_json(f, args.lower)
    source = ' '.join([' '.join(s) for s in source])
    tgt = ' '.join([' '.join([w for w in t if w != '@highlight']) for t in tgt])
    return {'docId': docId, 'src': source, 'tgt': tgt}


