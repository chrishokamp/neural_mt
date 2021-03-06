import os
import re
import argparse

import numpy as np

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
        # TODO: this interface needs to change for the new pickling format
	# TODO: the naming convention for blocks-internal pickling is also different -- they replace '/' with '-'
        # 17.4.16 -- why did the .get_value() API change?
        #param_value = model[param_name][()].get_value()
        param_value = model[param_name]
        output_name = re.sub('/', '.', param_name) + '.npz'

        if not os.path.isdir(extract_dir):
            os.mkdir(extract_dir)
        output_path = os.path.join(extract_dir, output_name)

        with open(output_path, 'wb') as output:
            np.save(output, param_value)

        print('saved model key: {} to {}'.format(param_name, output_path))

if __name__ == '__main__':
    config = vars(parser.parse_args())
    extract_model_parameters(config)
