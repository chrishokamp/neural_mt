"""
Count the number of unique words in a file
"""

import argparse
import codecs
import logging
from collections import Counter

from multiprocessing import Pool

logging.basicConfig()
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', help='the input corpus', required=True)

    args = parser.parse_args()

    vocab_counts = Counter()
    with codecs.open(args.input, encoding='utf8') as inp:
        for l in inp:
            vocab_counts.update(l.strip().split())

    logger.info(u'Stats for {}'.format(args.input))
    logger.info(u'Total vocab size: {}'.format(len(vocab_counts)))
    logger.info(u'Top 100 words: {}'.format(vocab_counts.most_common(100)))






