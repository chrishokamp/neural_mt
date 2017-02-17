"""
Create source and target vocabularies
"""

import logging
import argparse
import codecs
import itertools
import errno
import cPickle
from collections import Counter

from multiprocessing import Pool
import os

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def chunk_iterator(line_iterator, chunk_size=50000):
    while True:
        buffer = []
        for _ in range(chunk_size):
            try:
                buffer.append(line_iterator.next())
            except StopIteration:
                raise StopIteration
        yield buffer


def count_tokens(lines):
    src_toks = Counter()
    trg_toks = Counter()
    for source_line, target_line in lines:
        src_toks.update(source_line.strip().split())
        trg_toks.update(target_line.strip().split())

    return src_toks, trg_toks


def parallel_iterator(source_file, target_file):
    with codecs.open(source_file, encoding='utf8') as src:
        with codecs.open(target_file, encoding='utf8') as trg:
            for src_l, trg_l in itertools.izip(src, trg):
                yield (src_l, trg_l)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source",
                        help="the source text corpus")
    parser.add_argument("-t", "--target",
                        help="the target text corpus")
    parser.add_argument("-v", "--vocab", type=int,
                        help="the vocabulary size")
    parser.add_argument("-o", "--outputdir",
                        help="the directory where we should write the output")
    parser.add_argument("-sl", "--sourcelang",  help="the source language code")
    parser.add_argument("-tl", "--targetlang", help="the target language code")

    args = parser.parse_args()
    arg_dict = vars(args)

    chunk_factor = 500000
    parallel_chunk_iter = chunk_iterator(parallel_iterator(args.source, args.target), chunk_size=chunk_factor)

    num_procs = 10
    p = Pool(num_procs)

    source_vocab_counts = Counter()
    target_vocab_counts = Counter()
    # map and reduce
    for i, (src_count, trg_count) in enumerate(p.imap_unordered(count_tokens, parallel_chunk_iter)):
        source_vocab_counts += src_count
        target_vocab_counts += trg_count
        logger.info("{} lines processed".format(i*chunk_factor))

    source_vocab = [w for w, c in source_vocab_counts.most_common(args.vocab - 3)]
    target_vocab = [w for w, c in target_vocab_counts.most_common(args.vocab - 3)]

    source_vocab_len = min(args.vocab, len(source_vocab) + 1)
    target_vocab_len = min(args.vocab, len(target_vocab) + 1)

    # Special tokens and indexes
    source_default_mappings = [(u'<S>', 0), (u'<UNK>', 1), (u'</S>', source_vocab_len - 1)]
    target_default_mappings = [(u'<S>', 0), (u'<UNK>', 1), (u'</S>', target_vocab_len - 1)]
    source_vocab_dict = {k: v for k, v in source_default_mappings}
    target_vocab_dict = {k: v for k, v in target_default_mappings}

    for token, idx in zip(source_vocab, range(2, source_vocab_len - 1)):
        source_vocab_dict[token] = idx
    for token, idx in zip(target_vocab, range(2, target_vocab_len - 1)):
        target_vocab_dict[token] = idx

    # Now write the new files
    mkdir_p(args.outputdir)
    cPickle.dump(source_vocab_dict, open(os.path.join(args.outputdir, args.sourcelang + '.vocab.pkl'), 'w'))
    cPickle.dump(target_vocab_dict, open(os.path.join(args.outputdir, args.targetlang + '.vocab.pkl'), 'w'))

    logger.info('Wrote the vocabulary indexes to: {}'.format(args.outputdir))
    logger.info('Source Vocab size: {}'.format(source_vocab_len))
    logger.info('Target Vocab size: {}'.format(target_vocab_len))
    logger.info('Files in {}: {}'.format(args.outputdir, os.listdir(args.outputdir)))



