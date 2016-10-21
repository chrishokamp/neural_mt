"""Jointly shuffle two files"""
import logging
import argparse

from machine_translation.preprocessing_utils import shuffle_parallel

logging.basicConfig()
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--files', nargs='+', help='two text files of the same length to jointly shuffle',
                    required=True)
parser.add_argument('--temp_dir', type=str, default='./',
                    help='(Optional) temp directory where intermediate files will be stored')


if __name__ == '__main__':
    # shuffle the two files given at the command line in parallel
    args = parser.parse_args()
    logger.info("Jointly shuffling files: {}".format(args.files))
    temp_dir = args.temp_dir
    out_src, out_trg = shuffle_parallel(*args.files, temp_dir=temp_dir)
    logger.info("Wrote shuffled files to: {}".format([out_src, out_trg]))
