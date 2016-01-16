"""Jointly shuffle two files"""
import logging
import argparse

from neural_mt.machine_translation.preprocessing_utils import shuffle_parallel

logging.basicConfig()
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--files', nargs='2', help='two text files of the same length to jointly shuffle',
                    required=True)


if __name__ == '__main__':
    # shuffle the two files given at the command line in parallel
    args = parser.parse_args()
    logger.info("Jointly shuffling files: {}".format(args.files))
    out_src, out_trg = shuffle_parallel(*args.files)
    logger.info("Wrote shuffled files to: {}".format([out_src, out_trg]))
