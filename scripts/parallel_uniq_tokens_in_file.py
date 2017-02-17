
"""
Count the number of unique words in a large file in parallel
"""

import argparse
import codecs
import logging
from collections import Counter

from multiprocessing import Pool

logging.basicConfig()
logger = logging.getLogger(__name__)


def chunk_iter(iter, chunk_size):
    while True:
        try:
            chunk = (iter.next() for _ in range(chunk_size))
            yield chunk
        except StopIteration:
            raise


def count_words(lines):
    words_in_lines = Counter([w for l in lines for w in l.split()])
    return words_in_lines, len(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', help='the input corpus', required=True)

    args = parser.parse_args()
    outer_chunk_factor = 10000
    inner_chunk_factor = 1000
    num_procs = 10
    p = Pool(10)

    vocab_counts = Counter()
    with codecs.open(args.input, encoding='utf8') as inp:
        for ichunk in chunk_iter(inp, outer_chunk_factor):
            inner_iter = chunk_iter(ichunk, inner_chunk_factor)
            for c in p.map(count_words, inner_iter):
                vocab_counts += c

    logger.info(u'Stats for {}'.format(args.input))
    logger.info(u'Total vocab size: {}'.format(len(vocab_counts)))
    logger.info(u'Top 100 words: {}'.format(vocab_counts.most_common(100)))






