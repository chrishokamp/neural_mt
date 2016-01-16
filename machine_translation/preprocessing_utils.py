"""
This module is named 'preprocessing_utils' instead of 'preprocessing' to avoid conflicting
with the 'preprocessing' script in Groundhog
"""
import os
import uuid
import subprocess
import logging

from picklable_itertools.extras import equizip

logger = logging.getLogger(__name__)


def merge_parallel(src_filename, trg_filename, merged_filename):
    with open(src_filename, 'r') as left:
        with open(trg_filename, 'r') as right:
            with open(merged_filename, 'w') as final:
                for lline, rline in equizip(left, right):
                    if (lline != '\n') and (rline != '\n'):
                        final.write(lline[:-1] + ' ||| ' + rline)


def split_parallel(merged_filename, src_filename, trg_filename):
    with open(merged_filename) as combined:
        with open(src_filename, 'w') as left:
            with open(trg_filename, 'w') as right:
                for line in combined:
                    line = line.split('|||')
                    left.write(line[0].strip() + '\n')
                    right.write(line[1].strip() + '\n')


def shuffle_parallel(src_filename, trg_filename, temp_dir='./'):
    logger.info("Shuffling jointly [{}] and [{}]".format(src_filename,
                                                         trg_filename))
    out_src = src_filename + '.shuf'
    out_trg = trg_filename + '.shuf'
    merged_filename = os.path.join(temp_dir,str(uuid.uuid4()))
    shuffled_filename = os.path.join(temp_dir, str(uuid.uuid4()))
    if not os.path.exists(out_src) or not os.path.exists(out_trg):
        try:
            merge_parallel(src_filename, trg_filename, merged_filename)
            subprocess.check_call(
                " shuf {} > {} ".format(merged_filename, shuffled_filename),
                shell=True)
            split_parallel(shuffled_filename, out_src, out_trg)
            logger.info(
                "...files shuffled [{}] and [{}]".format(out_src, out_trg))
        except Exception as e:
            logger.error("{}".format(str(e)))
    else:
        logger.info("...files exist [{}] and [{}]".format(out_src, out_trg))
    if os.path.exists(merged_filename):
        os.remove(merged_filename)
    if os.path.exists(shuffled_filename):
        os.remove(shuffled_filename)
    return out_src, out_trg


