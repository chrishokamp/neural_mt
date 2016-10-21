"""
Merge the vocabularies in two or more cPickled dicts
    - by merge, we mean doing a union between the two
    - note that if the indexes in the vocabularies were constructed in some order, for example decreasing corpus
    frequency, that order will be lost
"""

import cPickle
import argparse
import logging
import os

import numpy

logging.basicConfig()
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-i','--inputidx', help='The input vocab index', required=True)
parser.add_argument('-t','--outputidx', help='The output vocab index', required=True)
parser.add_argument('-v','--vectors', help='The numpy array of vectors', required=True)
parser.add_argument('-o','--output', type=str, help='filename for the output vectors', required=True)


def unpickle(file):
    return cPickle.load(open(file))

def load_np_array(file):
    return numpy.load(open(file))

def map_index(input_vocab, output_vocab, input_vectors):
    """ Map the vector index indexed by input_index to the rows indexed by output_vocab

    """

    input_index = unpickle(input_vocab)
    output_index = unpickle(output_vocab)
    vectors = load_np_array(input_vectors)
    vector_len = vectors.shape[1]
    num_not_found = 0
    num_found = 0

    mapped_vectors = []
    for k, idx in sorted(output_index.items(), key=lambda x: x[1]):
        if k in input_index:
            mapped_vectors.append(vectors[input_index[k]])
            num_found += 1
        else:
            print('{} not found in input index'.format(k))
            num_not_found += 1
            mapped_vectors.append(numpy.random.normal(size=vector_len))

    print('total shared keys: {}'.format(num_found))
    print('total num indices not found: {}'.format(num_not_found))
    return numpy.vstack(mapped_vectors)


if __name__ == '__main__':
    args = parser.parse_args()

    logger.info("Mapping the tokens in {} to the vectors in {}, which are indexed by {}".format(args.outputidx,
                                                                                                args.inputidx,
                                                                                                args.vectors))
    new_vectors = map_index(args.inputidx, args.outputidx, args.vectors)

    if os.path.isfile(args.output):
        logger.warn("{} is already a file, but I'm going to overwrite it".format(args.output))
    with open(args.output + '.npz', 'wb') as out:
        numpy.save(out, new_vectors)
    logger.info("Saved the new index to: {}".format(args.output + '.npz'))






