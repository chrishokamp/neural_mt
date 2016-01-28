"""Encoder-Decoder with search for machine translation.

Please see `prepare_data.py` for preprocessing configuration

.. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
   Machine Translation by Jointly Learning to Align and Translate.
"""

import argparse
import logging
import pprint

import configurations

from machine_translation import main, NMTPredictor
from machine_translation.stream import get_tr_stream, get_dev_stream

from ipdb import set_trace


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
    logger.info("Model options:\n{}".format(pprint.pformat(config_obj)))

    if mode == 'train':
        # Get data streams and call main
        main(config_obj, get_tr_stream(**config_obj),
             get_dev_stream(**config_obj), args.bokeh)
    elif mode == 'predict':
        predictor = NMTPredictor(config_obj)
        predictor.predict_file(config_obj['test_set'], config_obj.get('translated_output_file', None))
        # TODO: Testing only
        # TODO: tokenize, preprocess, etc, etc
        # predictor.predict_sentence
    elif mode == 'server':

        import sys
        sys.path.append('.')
        from server import run_nmt_server

        # start restful server and log its port
        predictor = NMTPredictor(config_obj)
        run_nmt_server(predictor)


