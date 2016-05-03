"""
Functions and classes for sampling from NMT models

"""

import numpy


class SampleFunc:

    def __init__(self, sample_func, vocab):
        self.sample_func = sample_func
        self.vocab = vocab

    # TODO: we may be able to make this function faster by passing multiple sources for sampling at the same damn time
    # TODO: or by avoiding the for loop somehow
    def __call__(self, source_seq, num_samples=1):

        inputs = numpy.tile(source_seq[None, :], (num_samples, 1))
        # the output is [seq_len, batch]
        _1, outputs, _2, _3, costs = self.sample_func(inputs)
        outputs = outputs.T

        # TODO: this step could be avoided by computing the samples mask in a different way
        lens = self._get_true_length(outputs)
        samples = [s[:l] for s,l in zip(outputs.tolist(), lens)]

        return samples

    def _get_true_length(self, seqs):
        try:
            lens = []
            for r in seqs.tolist():
                lens.append(r.index(self.vocab['</S>']) + 1)
            return lens
        except ValueError:
            return [seqs.shape[1] for _ in range(seqs.shape[0])]




