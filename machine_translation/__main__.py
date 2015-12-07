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
parser.add_argument("--proto",  default="get_config",
                    help="Prototype config to use for config")
parser.add_argument("--datadir",  default="./data/",
                    help="The directory where data should be stored")
parser.add_argument("--bokeh",  default=False, action="store_true",
                    help="Use bokeh server for plotting")
args = parser.parse_args()


if __name__ == "__main__":
    # Get configurations for model
    arg_dict = vars(parser.parse_args())
    print('arg_dict')
    print(arg_dict)
    configuration = getattr(configurations, args.proto)(arg_dict['datadir'])
    logger.info("Model options:\n{}".format(pprint.pformat(configuration)))
    # Get data streams and call main
    main(configuration, get_tr_stream(**configuration),
         get_dev_stream(**configuration), args.bokeh)
