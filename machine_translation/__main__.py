"""Encoder-Decoder with search for machine translation.

Please see `prepare_data.py` for preprocessing configuration

.. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
   Machine Translation by Jointly Learning to Align and Translate.
"""

import argparse
import logging
import pprint

import configurations

from machine_translation import main
from machine_translation.stream import get_tr_stream, get_dev_stream

logger = logging.getLogger(__name__)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("exp_config", required=True,
                    help="Path to the yaml config file for your experiment")
parser.add_argument("--bokeh",  default=False, action="store_true",
                    help="Use bokeh server for plotting")
args = parser.parse_args()


if __name__ == "__main__":
    # Get configurations for model
    args = parser.parse_args()
    arg_dict = vars(args)
    configuration_file = arg_dict['exp_config']
    config_obj = configurations.get_config(configuration_file)
    logger.info("Model options:\n{}".format(pprint.pformat(config_obj)))
    # Get data streams and call main
    main(config_obj, get_tr_stream(**config_obj),
         get_dev_stream(**config_obj), args.bokeh)
