import os
import re
import argparse
import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(
    description="""Extract the parameters from a serialized blocks model""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("model_file", type=str, help="The location of your serialized .npz model")
    parser.add_argument("extract_dir", type=str, help="Where to save the output files")


# the model _MUST_ be loaded in the same python context (same modules available), otherwise there will be an error
# also, if the model was trained on the GPU, the saved model must be loaded on the GPU
def extract_model_parameters(config):

    model = np.load(config['model_file'])
    extract_dir = config['extract_dir']

    for param_name in model.keys():
        param_value = model[param_name][()].get_value()
        output_name = re.sub('/', '.', param_name) + '.npz'
        output_path = os.path.join(extract_dir, output_name)

        with open(output_path, 'wb') as output:
            np.save(output, param_value)

    logger.info('saved model key: {} to {}'.format(param_name, output_name))

if __name__ == '__main__':
    config = vars(parser.parse_args())
    extract_model_parameters(config)
