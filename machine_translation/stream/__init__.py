import numpy
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping, Transformer)

from six.moves import cPickle


def _ensure_special_tokens(vocab, bos_idx=0, eos_idx=0, unk_idx=1):
    """Ensures special tokens exist in the dictionary."""

    # remove tokens if they exist in some other index
    tokens_to_remove = [k for k, v in vocab.items()
                        if v in [bos_idx, eos_idx, unk_idx]]
    for token in tokens_to_remove:
        vocab.pop(token)
    # put corresponding item
    # note the BOS, EOS, and UNK tokens are hard-coded here
    vocab['<S>'] = bos_idx
    vocab['</S>'] = eos_idx
    vocab['<UNK>'] = unk_idx
    return vocab


def _length(sentence_pair):
    """Assumes target is the second element in the tuple."""
    return len(sentence_pair[1])


class PaddingWithEOS(Padding):
    """Padds a stream with given end of sequence idx."""
    def __init__(self, data_stream, eos_idx, **kwargs):

        self.suffix_length = None
        if 'suffix_length' in kwargs:
            self.suffix_length = kwargs.pop('suffix_length')
            try:
                self.truncate_sources = set(kwargs.pop('truncate_sources'))
            except KeyError as e:
                raise(e) # you must specify which sources you are truncating with self.suffix_length

        kwargs['data_stream'] = data_stream
        # note that eos_idx is a list containing the eos_idx for each source that will be padded
        self.eos_idx = eos_idx
        super(PaddingWithEOS, self).__init__(**kwargs)

    def transform_batch(self, batch):
        batch_with_masks = []
        for i, (source, source_data) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                batch_with_masks.append(source_data)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_data]
            lengths = [shape[0] for shape in shapes]
            max_sequence_length = max(lengths)
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_data[0]).dtype

            padded_data = numpy.ones(
                (len(source_data), max_sequence_length) + rest_shape,
                dtype=dtype) * self.eos_idx[i]
            for i, sample in enumerate(source_data):
                padded_data[i, :len(sample)] = sample
            batch_with_masks.append(padded_data)

            mask = numpy.zeros((len(source_data), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                if self.suffix_length is not None and source in self.truncate_sources:
                    mask[i, :self.suffix_length] = 1.
                else:
                    mask[i, :sequence_length] = 1.
            batch_with_masks.append(mask)
        return tuple(batch_with_masks)


class _oov_to_unk(object):
    """Maps out of vocabulary token index to unk token index."""
    def __init__(self, src_vocab_size=30000, trg_vocab_size=30000,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.unk_id = unk_id

    def __call__(self, sentence_pair):
        return ([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[0]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[1]])


class _too_long(object):
    """Filters sequences longer than given sequence length."""
    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return all([len(sentence) <= self.seq_len
                    for sentence in sentence_pair])


def get_tr_stream(src_vocab, trg_vocab, src_data, trg_data,
                  src_vocab_size=30000, trg_vocab_size=30000, unk_id=1,
                  seq_len=50, batch_size=80, sort_k_batches=12, bos_token=None, **kwargs):
    """Prepares the training data stream."""
    if type(bos_token) is str:
        bos_token = bos_token.decode('utf8')

    # Load dictionaries and ensure special tokens exist
    src_vocab = _ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else cPickle.load(open(src_vocab)),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
    trg_vocab = _ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        cPickle.load(open(trg_vocab)),
        bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

    # Get text files from both source and target
    src_dataset = TextFile([src_data], src_vocab,
                           bos_token=bos_token,
                           eos_token=u'</S>',
                           unk_token=u'<UNK>',
                           encoding='utf8')
    trg_dataset = TextFile([trg_data], trg_vocab,
                           bos_token=bos_token,
                           eos_token=u'</S>',
                           unk_token=u'<UNK>',
                           encoding='utf8')

    # Merge them to get a source, target pair
    stream = Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))

    # Filter sequences that are too long
    stream = Filter(stream,
                    predicate=_too_long(seq_len=seq_len))

    # Replace out of vocabulary tokens with unk token
    # TODO: doesn't the TextFile stream do this anyway?
    stream = Mapping(stream,
                     _oov_to_unk(src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

    # Now make a very big batch that we can shuffle
    shuffle_batch_size = kwargs.get('shuffle_batch_size', 1000)
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(shuffle_batch_size)
                   )

    stream = ShuffleBatchTransformer(stream)

    # unpack it again
    stream = Unpack(stream)
    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(
                       batch_size*sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Construct batches from the stream with specified batch size
    stream = Batch(
        stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    masked_stream = PaddingWithEOS(
        stream, [src_vocab_size - 1, trg_vocab_size - 1])

    return masked_stream, src_vocab, trg_vocab


def get_dev_stream(val_set=None, src_vocab=None, src_vocab_size=30000,
                   unk_id=1, bos_token=None, **kwargs):
    """Setup development set stream if necessary."""
    if type(bos_token) is str:
        bos_token = bos_token.decode('utf8')

    dev_stream = None
    if val_set is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab)),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        dev_dataset = TextFile([val_set], src_vocab,
                               bos_token=bos_token,
                               eos_token=u'</S>',
                               unk_token=u'<UNK>',
                               encoding='utf8')

        dev_stream = DataStream(dev_dataset)
    return dev_stream


# Module for functionality associated with streaming data
class MTSampleStreamTransformer:
    """
    Stateful transformer which takes a stream of (source, target) and adds the sources ('samples', 'scores')

    Samples are generated by calling the sample func with the source as argument

    Scores are generated by comparing each generated sample to the reference

    Parameters
    ----------
    sample_func: function(num_samples=1) which takes source seq and outputs <num_samples> samples
    score_func: function

    At call time, we expect a stream providing (sources, references) -- i.e. something like a TextFile object


    """

    def __init__(self, sample_func, score_func, num_samples=1, **kwargs):
        self.sample_func = sample_func
        self.score_func = score_func
        self.num_samples = num_samples
        # kwargs will get passed to self.score_func when it gets called
        self.kwargs = kwargs

    def __call__(self, data, **kwargs):
        source = data[0]
        reference = data[1]

        # each sample may be of different length
        samples = self.sample_func(numpy.array(source), self.num_samples)
        # TODO: we currently have to pass the source because of the interface to mteval_v13
        scores = numpy.array(self._compute_scores(source, reference, samples, **self.kwargs)).astype('float32')

        return (samples, scores)

    # Note that many sentence-level metrics like BLEU can be computed directly over the indexes (not the strings),
    # Note that some sentence-level metrics like METEOR require the string representation
    # if the scoring function needs to map from ints to strings, provide 'src_vocab' and 'trg_vocab' via the kwargs
    # So we don't need to map back to a string representation
    def _compute_scores(self, source, reference, samples, **kwargs):
        """Call the scoring function to compare each sample to the reference"""

        return self.score_func(source, reference, samples, **kwargs)


class FlattenSamples(Transformer):
    """Look for a source called 'samples' in the data_stream, flatten the array of (batch_size, num_samples) to
    (batch_size * num_samples)

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to wrap

    """
    def __init__(self, data_stream, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')
        super(FlattenSamples, self).__init__(
            data_stream, produces_examples=False, **kwargs)

#         if mask_dtype is None:
#             self.mask_dtype = config.floatX
#         else:
#             self.mask_dtype = mask_dtype

    @property
    def sources(self):
        return self.data_stream.sources

    def transform_batch(self, batch):
        batch_with_flattened_samples = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):

            if source == 'samples' or source == 'seq_probs':
                flattened_samples = []
                for ins in source_batch:
                    for sample in ins:
                        flattened_samples.append(sample)
                batch_with_flattened_samples.append(flattened_samples)
            else:
                batch_with_flattened_samples.append(source_batch)

        return tuple(batch_with_flattened_samples)


class CopySourceNTimes(Transformer):
    """Duplicate the source N times to match the number of samples

    We need this transformer because the attention model expects one source sequence for each
    target sequence, but in the sampling case there are effectively (instances*sample_size) target sequences

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to wrap
    n_samples : int -- the number of samples that were generated for each source sequence

    """
    def __init__(self, data_stream, n_samples=5, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')
        self.n_samples = n_samples

        super(CopySourceNTimes, self).__init__(
            data_stream, produces_examples=False, **kwargs)


    @property
    def sources(self):
        return self.data_stream.sources

    def transform_batch(self, batch):
        batch_with_expanded_source = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source == 'source':
                # copy each source sequence self.n_samples times, but keep the tensor 2d
                expanded_source = []
                for ins in source_batch:
                    expanded_source.extend([ins for _ in range(self.n_samples)])

                batch_with_expanded_source.append(expanded_source)
            else:
                batch_with_expanded_source.append(source_batch)

        return tuple(batch_with_expanded_source)


class ShuffleBatchTransformer(Transformer):
    """
    Take batches and shuffle them, allow user to provide the random seed

    This transformer can be used to shuffle datastreams of an unknown or infinite size,
    in other words data streams which are not indexable


    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to wrap
    # TODO: what is the right way to pass a random seed
    random_seed : int -- the number of samples that were generated for each source sequence

    """
    def __init__(self, data_stream, random_seed=None, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')

        super(ShuffleBatchTransformer, self).__init__(
            data_stream, produces_examples=False, **kwargs)


    @property
    def sources(self):
        return self.data_stream.sources

    # TODO: add user-configurable random seed
    # TODO: add test for this function
    def transform_batch(self, batch):
        assert len(set([len(a) for a in batch])) == 1, 'the first dimension of all sources must be the same'

        batch = [numpy.array(a) for a in zip(*numpy.random.permutation(zip(*batch)))]

        return tuple(batch)


def get_textfile_stream(source_file=None, src_vocab=None, src_vocab_size=30000,
                      unk_id=1, bos_token=None):
    """Create a TextFile dataset from a single text file, and return a stream"""
    if type(bos_token) is str:
        bos_token = bos_token.decode('utf8')

    src_vocab = _ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict) else
        cPickle.load(open(src_vocab)),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
    source_dataset = TextFile([source_file], src_vocab,
                              bos_token=bos_token,
                              eos_token=u'</S>',
                              unk_token=u'<UNK>',
                              encoding='utf8')
    source_stream = source_dataset.get_example_stream()
    return source_stream



