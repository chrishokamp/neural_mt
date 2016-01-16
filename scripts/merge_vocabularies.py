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

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-f','--files', nargs='+', help='List of paths to cpickled vocabulary dicts', required=True)
parser.add_argument('-o','--output', type=str, help='filename for the output', required=True)


def unpickle(file):
    return cPickle.load(open(file))

def merge_dicts(dicts):
    """"merge dict keys, but don't duplicate the values
        - This implementation depends upon the values of the first dict being ascending integers
        - first create the set of all keys that are not in the first dict, then get the max value of the first dict
        - add the keys to the first dict, with values starting from max+1

    """

    first_dict_keys = set(dicts[0].keys())
    max_value = max(dicts[0].values())
    other_dict_keys = set([k for d in dicts[1:] for k in d.keys()])
    new_keys = other_dict_keys.difference(first_dict_keys)
    num_new_keys = len(new_keys)

    merged = dicts[0]
    merged.update([(k,v) for k,v in zip(new_keys, range(max_value+1, max_value+num_new_keys+1))])
    import ipdb;ipdb.set_trace()

    # check that we have the right number of total keys
    assert len(merged.keys()) == num_new_keys + len(first_dict_keys)
    # check that we have unique values for all of the keys

    if len(set(merged.values())) != len(merged.keys()):
        logger.warn('The total number of keys is: {}, but the total number of indexes is: {}'
                    .format(merged.keys()), merged.values())

    return merged


if __name__ == '__main__':
    args = parser.parse_args()

    logger.info("Merging the vocabularies at: {}".format(args.files))
    dicts = [cPickle.load(open(f)) for f in args.files]

    merged = merge_dicts(dicts)

    if os.path.isfile(args.output):
        logger.warn("{} is already a file, but I'm going to overwrite it".format(args.output))
    with open(args.output, 'wb') as out:
        cPickle.dump(merged, out)

    logger.info("Saved the merged vocabularies to: {}".format(args.output))






