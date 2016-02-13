#!/usr/bin/python

import argparse
import logging
import os
import subprocess
import tarfile
import urllib2
import uuid

from preprocessing_utils import merge_parallel, split_parallel, shuffle_parallel

TRAIN_DATA_URL = 'http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz'
VALID_DATA_URL = 'http://www.statmt.org/wmt15/dev-v2.tgz'
PREPROCESS_URL = 'https://raw.githubusercontent.com/lisa-groundhog/' +\
                 'GroundHog/master/experiments/nmt/preprocess/preprocess.py'
TOKENIZER_URL = 'https://raw.githubusercontent.com/moses-smt/mosesdecoder/' +\
                'master/scripts/tokenizer/tokenizer.perl'

TOKENIZER_PREFIXES = 'https://raw.githubusercontent.com/moses-smt/' +\
                     'mosesdecoder/master/scripts/share/nonbreaking_' +\
                     'prefixes/nonbreaking_prefix.'
BLEU_SCRIPT_URL = 'https://raw.githubusercontent.com/moses-smt/mosesdecoder' +\
                  '/master/scripts/generic/multi-bleu.perl'

# these get overwritten when command line args are read
OUTPUT_DIR = ''
PREFIX_DIR = ''

parser = argparse.ArgumentParser(
    description="""
This script donwloads parallel corpora given source and target pair language
indicators and preprocess it respectively for neural machine translation.For
the preprocessing, moses tokenizer is applied first then tokenized corpora
are used to extract vocabularies for source and target languages. Finally the
tokenized parallel corpora are shuffled for SGD.

Note that the default behavior of this script is written specifically for
the WMT15 training and development corpora, hence change the corresponding sections if you plan to use
some other data.
""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-s", "--source", type=str, help="Source language",
                    default="cs")
parser.add_argument("-t", "--target", type=str, help="Target language",
                    default="en")
parser.add_argument("--source-dev", type=str, default="newstest2013.cs",
                    help="Source language dev filename")
parser.add_argument("--target-dev", type=str, default="newstest2013.en",
                    help="Target language dev filename")
parser.add_argument("--source-vocab", type=int, default=30000,
                    help="Source language vocabulary size")
parser.add_argument("--target-vocab", type=int, default=30000,
                    help="Target language vocabulary size")
parser.add_argument("--data_dir", type=str, default='./data/',
                    help="Directory where the data will be stored")
parser.add_argument("--prefix_dir", type=str, default='./share/nonbreaking_prefixes',
                    help="Directory where the prefix lists will be stored")
parser.add_argument("--threads", type=int, default=1,
                    help="The number of threads available")
parser.add_argument("--train_data", type=str, default=None,
                    help="Optional path to user-specified training data in .tgz, or a directory containing text files")
parser.add_argument("--dev_data", type=str, default=None,
                    help="Optional path to user-specified dev data in .tgz, or a directory containing text files " +
                         "(overrides source-dev and target-dev).")
parser.add_argument("--source_vocab_file", type=str, default=None,
                    help="Optional path to a file to use to create the source vocabulary")
parser.add_argument("--target_vocab_file", type=str, default=None,
                    help="Optional path to a file to use to create the target vocabulary")


def download_and_write_file(url, file_name):
    logger.info("Downloading [{}]".format(url))
    if not os.path.exists(file_name):
        path = os.path.dirname(file_name)
        if not os.path.exists(path):
            os.makedirs(path)
        u = urllib2.urlopen(url)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        logger.info("...saving to: %s Bytes: %s" % (file_name, file_size))
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % \
                (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print status,
        f.close()
    else:
        logger.info("...file exists [{}]".format(file_name))


# TODO: names_to_look should be optional -- only in the case where user doesn't explictly specify the parallel data
def find_lang_pair_files_in_tar(file_to_extract, names_to_look):
    try:
        tar = tarfile.open(file_to_extract, 'r')
        src_trg_files = [ff for ff in tar.getnames()
                         if any([ff.find(nn) > -1 for nn in names_to_look])]
        if not len(src_trg_files):
            raise ValueError("[{}] pair does not exist in the archive!"
                             .format(src_trg_files))
        return src_trg_files
    except Exception as e:
        logger.error("{}".format(str(e)))


def extract_tar(file_to_extract, extract_into, files_to_extract=None):

    tar = tarfile.open(file_to_extract, 'r')
    # if user didn't specify which files to extract, extract everything
    if files_to_extract is None:
        files_to_extract = tar.getnames()

    extracted_filenames = []
    for item in tar:
        # extract only source-target pair
        if item.name in files_to_extract:
            logger.info("Extracting file [{}] into [{}]"
                        .format(file_to_extract, extract_into))
            file_path = os.path.join(extract_into, item.path)
            if not os.path.exists(file_path):
                logger.info("...extracting [{}] into [{}]"
                            .format(item.name, file_path))
                tar.extract(item, extract_into)
                extracted_filenames.append(os.path.join(extract_into, item.path))
            else:
                logger.info("...file exists [{}]".format(file_path))
                extracted_filenames.append(os.path.join(extract_into, item.path))
    return extracted_filenames


def tokenize_text_files(files_to_tokenize, tokenizer, threads=1):
    for name in files_to_tokenize:
        logger.info("Tokenizing file [{}]".format(name))
        out_file = os.path.join(
            OUTPUT_DIR, os.path.basename(name) + '.tok')
        logger.info("...writing tokenized file [{}]".format(out_file))

        # we get the language for the tokenizer from the final element of the filename (after the last '.')
        tokenizer_command = ["perl", tokenizer,  "-l", name.split('.')[-1], "-threads", str(threads), "-no-escape", "1"]
        if not os.path.exists(out_file):
            with open(name, 'r') as inp:
                with open(out_file, 'w', 0) as out:
                    subprocess.check_call(
                        tokenizer_command, stdin=inp, stdout=out, shell=False)
        else:
            logger.info("...file exists [{}]".format(out_file))


def create_vocabularies(tr_files, preprocess_file, source_vocab_file=None, target_vocab_file=None):
    """
    :param tr_files:
    :param preprocess_file:
    :return:

    Note this function depends on some tr_files ending with args.source and args.target
    """
    src_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            args.source, args.target, args.source))
    trg_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            args.source, args.target, args.target))

    # note that the files are required to end in <LANG_CODE>.tok -- eventually we should make this more flexible
    src_filename = os.path.basename(
        tr_files[[i for i, n in enumerate(tr_files)
                  if n.endswith(args.source)][0]]) + '.tok'
    trg_filename = os.path.basename(
        tr_files[[i for i, n in enumerate(tr_files)
                  if n.endswith(args.target)][0]]) + '.tok'

    if source_vocab_file is None:
        source_vocab_file = src_filename
    if target_vocab_file is None:
        target_vocab_file = trg_filename

    logger.info("Creating source vocabulary [{}]".format(src_vocab_name))
    if not os.path.exists(src_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, src_vocab_name, args.source_vocab,
            os.path.join(OUTPUT_DIR, source_vocab_file)),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(src_vocab_name))

    logger.info("Creating target vocabulary [{}]".format(trg_vocab_name))
    if not os.path.exists(trg_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, trg_vocab_name, args.target_vocab,
            os.path.join(OUTPUT_DIR, target_vocab_file)),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(trg_vocab_name))
    return src_filename, trg_filename




def main(arg_dict):
    # if user didn't specify the training data, do the default (assume .tgz file containing two files (src and tgt)
    # note that files must end with the lowercase language suffix -- i.e. *.en

    # scripts that we'll pull from the moses and groundhog githubs
    preprocess_file = os.path.join(OUTPUT_DIR, 'preprocess.py')
    bleuscore_file = os.path.join(OUTPUT_DIR, 'multi-bleu.perl')
    tokenizer_file = os.path.join(OUTPUT_DIR, 'tokenizer.perl')
    source_prefix_file = os.path.join(PREFIX_DIR,
                                      'nonbreaking_prefix.' + args.source)
    target_prefix_file = os.path.join(PREFIX_DIR,
                                      'nonbreaking_prefix.' + args.target)
    # Download bleu score calculation script
    download_and_write_file(BLEU_SCRIPT_URL, bleuscore_file)

    # Download preprocessing script
    download_and_write_file(PREPROCESS_URL, preprocess_file)

    # Download tokenizer
    download_and_write_file(TOKENIZER_URL, tokenizer_file)
    download_and_write_file(TOKENIZER_PREFIXES + args.source,
                            source_prefix_file)
    download_and_write_file(TOKENIZER_PREFIXES + args.target,
                            target_prefix_file)

    source_lang = arg_dict['source']
    target_lang = arg_dict['target']

    user_train = arg_dict.get('train_data', None)
    user_dev = arg_dict.get('dev_data', None)
    output_dir = arg_dict['data_dir']

    # training data
    if user_train is None:
        train_data_file = os.path.join(OUTPUT_DIR, 'tmp', 'train_data.tgz')
        # Download the News Commentary v10 ~122Mb and extract it
        download_and_write_file(TRAIN_DATA_URL, train_data_file)
        # if user wants just the news-commentary data
        train_files_to_extract = find_lang_pair_files_in_tar(
            train_data_file, ["{}-{}".format(args.source, args.target)])
        training_files = extract_tar(train_data_file, os.path.dirname(train_data_file), train_files_to_extract)
    else:
        # use user data
        logger.info('Using training data at: {}'.format(user_train))

        if user_train.endswith('.tgz'):
            training_files = extract_tar(user_train, os.path.dirname(output_dir))
        elif os.path.isdir(user_train):
            training_files = [os.path.join(user_train, f) for f in os.listdir(user_train)
                              if f.endswith((source_lang, target_lang))]
        else:
            raise ValueError("you need to pass a .tgz file or a directory name as --train_data")

    # dev data
    if user_dev is None:
        valid_data_file = os.path.join(output_dir, 'tmp', 'valid_data.tgz')
        # Download development set and extract it
        download_and_write_file(VALID_DATA_URL, valid_data_file)
        valid_files_to_extract = find_lang_pair_files_in_tar(
            valid_data_file, [args.source_dev, args.target_dev])
        validation_files = extract_tar(valid_data_file, os.path.dirname(valid_data_file), valid_files_to_extract)
    else:
        # use user's dev data
        logger.info('Using dev data at: {}'.format(user_dev))

        if user_dev.endswith('.tgz'):
            validation_files = extract_tar(user_dev, os.path.dirname(output_dir))
        elif os.path.isdir(user_dev):
            validation_files = [os.path.join(user_dev, f) for f in os.listdir(user_dev)
                                if f.endswith((source_lang, target_lang))]
        else:
            raise ValueError("you need to pass a .tgz file or a directory name as --dev_data")

    # Apply tokenizer
    threads = arg_dict.get('threads', 1)

    # TODO: make each step optional

    # step 1 - tokenize train and dev
    # tokenize_text_files closes over OUTPUT_DIR, set in __main__ - same value as output_dir and args['data_dir']
    tokenize_text_files(training_files + validation_files, tokenizer_file, threads=threads)


    # step 2 - create vocabularies from training data
    # Apply preprocessing and construct vocabularies
    # TODO: create_vocabularies does two things: (1) writes the vocabulary .pkls to disk, 2 returns the filenames
    # TODO: the source and target filenames already exist from the tokenization step, so they should be returned from there
    # optional files to use for the vocab creation -- useful if you want to filter the MT vocab by the words contained
    # in another file
    src_vocab_file = arg_dict['source_vocab_file']
    tgt_vocab_file = arg_dict['target_vocab_file']

    src_filename, trg_filename = create_vocabularies(training_files, preprocess_file,
                                                     source_vocab_file=src_vocab_file,
                                                     target_vocab_file=tgt_vocab_file)

    # step 3 - Shuffle the training datasets
    shuffle_parallel(os.path.join(OUTPUT_DIR, src_filename),
                     os.path.join(OUTPUT_DIR, trg_filename), temp_dir=OUTPUT_DIR)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('prepare_data')

    # NOTE -- IMPORTANT: Some of the functions close over args
    args = parser.parse_args()
    arg_dict = vars(args)
    OUTPUT_DIR = arg_dict['data_dir']
    PREFIX_DIR = arg_dict['prefix_dir']
    main(arg_dict)
