"""Encoder-Decoder with search for machine translation.

Please see `prepare_data.py` for preprocessing configuration

.. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
   Machine Translation by Jointly Learning to Align and Translate.
"""

from __future__ import print_function
import argparse
import logging
import pprint
import codecs
import re
import os
import time
from subprocess import Popen, PIPE

import configurations

from machine_translation import main, NMTPredictor
from machine_translation.stream import get_tr_stream, get_dev_stream

logging.basicConfig()
logger = logging.getLogger(__name__)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("exp_config",
                    help="Path to the yaml config file for your experiment")
parser.add_argument("-m", "--mode", default='train',
                    help="The mode we are in [train,predict,server] -- default=train")
parser.add_argument("--bokeh",  default=False, action="store_true",
                    help="Use bokeh server for plotting")

if __name__ == "__main__":
    # Get configurations for model
    args = parser.parse_args()
    arg_dict = vars(args)
    configuration_file = arg_dict['exp_config']
    mode = arg_dict['mode']
    logger.info('Running Neural Machine Translation in mode: {}'.format(mode))
    config_obj = configurations.get_config(configuration_file)
    # add the config file name into config_obj
    config_obj['config_file'] = configuration_file
    logger.info("Model Configuration:\n{}".format(pprint.pformat(config_obj)))

    if mode == 'train':
        # Get data streams and call main
        main(config_obj, get_tr_stream(**config_obj),
             get_dev_stream(**config_obj), args.bokeh)
    elif mode == 'predict':
        predictor = NMTPredictor(config_obj)
        predictor.predict_file(config_obj['test_set'], config_obj.get('translated_output_file', None))
    elif mode == 'evaluate':
        logger.info("Started Evaluation: ")
        val_start_time = time.time()
        # TODO: support more evaluation metrics than just BLEU score

        # translate if necessary, write output file, call external evaluation tools and show output
        translated_output_file = config_obj.get('translated_output_file', None)
        if translated_output_file is not None and os.path.isfile(translated_output_file):
                logger.info('{} already exists, so I\'m evaluating the BLEU score of this file with respect to the ' +
                            'reference that you provided: {}'.format(translated_output_file,
                                                                     config_obj['test_gold_refs']))
        else:
            predictor = NMTPredictor(config_obj)
            logger.info('Translating: {}'.format(config_obj['test_set']))
            translated_output_file = predictor.predict_file(config_obj['test_set'],
                                                            translated_output_file)
            logger.info('Translated: {}, output was written to: {}'.format(config_obj['test_set'],
                                                                           translated_output_file))

        # get gold refs
        multibleu_cmd = ['perl', config_obj['bleu_script'],
                         config_obj['test_gold_refs'], '<']

        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE)

        with codecs.open(translated_output_file, encoding='utf8') as hyps:
            for l in hyps.read().strip().split('\n'):
                # send the line to the BLEU script
                print(l.encode('utf8'), file=mb_subprocess.stdin)

        mb_subprocess.stdin.flush()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logger.info(stdout)
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        logger.info('BLEU SCORE: {}'.format(bleu_score))
        mb_subprocess.terminate()

    elif mode == 'server':

        import sys
        sys.path.append('.')
        from server import run_nmt_server

        # start restful server and log its port
        predictor = NMTPredictor(config_obj)
        run_nmt_server(predictor)


