# coding: utf-8

import os
import codecs
import subprocess
from pprint import pprint
from subprocess import Popen, PIPE, STDOUT

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

import numpy
import codecs
import tempfile
import cPickle
import copy
from collections import OrderedDict
import itertools
from theano import tensor

from fuel.datasets import Dataset
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)
from fuel.transformers import Transformer

from blocks.algorithms import (GradientDescent, StepClipping,
                               CompositeRule, Adam, AdaDelta)
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector
from blocks.search import BeamSearch
from blocks_extras.extensions.plot import Plot

from machine_translation.checkpoint import CheckpointNMT, LoadNMT
from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.sampling import BleuValidator, Sampler, SamplingBase
from machine_translation.stream import (get_tr_stream, get_dev_stream,
                                        _ensure_special_tokens, MTSampleStreamTransformer,
                                        get_textfile_stream, _too_long, _length, PaddingWithEOS,
                                        _oov_to_unk)
from machine_translation.evaluation import sentence_level_bleu

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False



# build the training and sampling graphs for minimum risk training
# Intialize the MTSampleStreamTransformer with the sampling function

# load a model that's already trained, and start tuning it with minimum-risk
# mock-up training using the blocks main loop

# TODO: Integrate configuration so min-risk training is a single line in the config file
# TODO: this requires handling both the data stream and the Model cost function

# create the graph which can sample from our model
# Note that we must sample instead of getting the 1-best or N-best, because we need the randomness to make the expected
# BLEU score make sense

BASEDIR = '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BERTHA-TEST_Adam_wmt-multimodal_internal_data_dropout'+\
          '0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/'
#best_bleu_model_1455464992_BLEU31.61.npz

exp_config = {
    'src_vocab_size': 20000,
    'trg_vocab_size': 20000,
    'enc_embed': 300,
    'dec_embed': 300,
    'enc_nhids': 800,
    'dec_nhids': 800,
    'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BERTHA-TEST_wmt-multimodal_internal_data_dropout0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/best_bleu_model_1455410311_BLEU30.38.npz',
    'src_vocab': os.path.join(BASEDIR, 'vocab.en-de.en.pkl'),
    'trg_vocab': os.path.join(BASEDIR, 'vocab.en-de.de.pkl'),
    'src_data': os.path.join(BASEDIR, 'training_data/train.en.tok.shuf'),
    'trg_data': os.path.join(BASEDIR, 'training_data/train.de.tok.shuf'),
    'unk_id':1,
    # Bleu script that will be used (moses multi-perl in this case)
    'bleu_script': os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '../test_data/sample_experiment/tiny_demo_dataset/multi-bleu.perl'),

    # Optimization related ----------------------------------------------------
    # Batch size
    'batch_size': 8,
    # This many batches will be read ahead and sorted
    'sort_k_batches': 2,
    # Optimization step rule
    'step_rule': 'AdaDelta',
    # Gradient clipping threshold
    'step_clipping': 1.,
    # Std of weight initialization
    'weight_scale': 0.01,
    'seq_len': 40,
    # Beam-size
    'beam_size': 10,

    # Maximum number of updates
    'finish_after': 1000000,

    # Reload model from files if exist
    'reload': False,

    # Save model after this many updates
    'save_freq': 500,

    # Show samples from model after this many updates
    'sampling_freq': 1000,

    # Show this many samples at each sampling
    'hook_samples': 5,

    # Validate bleu after this many updates
    'bleu_val_freq': 10,
    # Normalize cost according to sequence length after beam-search
    'normalized_bleu': True,
    
    'saveto': '/media/1tb_drive/test_min_risk_model_save',
    'model_save_directory': 'test_min_risk_model_save',
    # Validation set source file
    'val_set': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/dev.en.tok',

    # Validation set gold file
    'val_set_grndtruth': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/dev.de.tok',

    # Print validation output to file
    'output_val_set': True,

    # Validation output file
    'val_set_out': '/media/1tb_drive/test_min_risk_model_save/validation_out.txt',
    'val_burn_in': 0,

    # NEW PARAM FOR MIN RISK
    'n_samples': 100

}


def get_sampling_model_and_input(exp_config):
    # Create Theano variables
    encoder = BidirectionalEncoder(
        exp_config['src_vocab_size'], exp_config['enc_embed'], exp_config['enc_nhids'])

    decoder = Decoder(
        exp_config['trg_vocab_size'], exp_config['dec_embed'], exp_config['dec_nhids'],
        exp_config['enc_nhids'] * 2, loss_function='min_risk')

    # Create Theano variables
    logger.info('Creating theano variables')
    sampling_input = tensor.lmatrix('source')

    # Get beam search
    logger.info("Building sampling model")
    sampling_representation = encoder.apply(
        sampling_input, tensor.ones(sampling_input.shape))
    generated = decoder.generate(sampling_input, sampling_representation)

#     _, samples = VariableFilter(
#         bricks=[decoder.sequence_generator], name="outputs")(
#                  ComputationGraph(generated[1]))  # generated[1] is next_outputs
#     beam_search = BeamSearch(samples=samples)

    # build the model that will let us get a theano function from the sampling graph
    logger.info("Creating Sampling Model...")
    sampling_model = Model(generated)


    return sampling_model, sampling_input, encoder, decoder

sample_model, theano_sampling_input, train_encoder, train_decoder = get_sampling_model_and_input(exp_config)


# test that we can pull samples from the model
trg_vocab = cPickle.load(open(exp_config['trg_vocab']))
trg_vocab_size = exp_config['trg_vocab_size'] - 1
src_vocab = cPickle.load(open(exp_config['src_vocab']))
src_vocab_size = exp_config['src_vocab_size'] - 1

src_vocab = _ensure_special_tokens(src_vocab, bos_idx=0,
                                   eos_idx=src_vocab_size, unk_idx=exp_config['unk_id'])
trg_vocab = _ensure_special_tokens(trg_vocab, bos_idx=0,
                                   eos_idx=trg_vocab_size, unk_idx=exp_config['unk_id'])

theano_sample_func = sample_model.get_theano_function()

# note: we close over the sampling func and the trg_vocab to standardize the interface
# TODO: actually this should be a callable class with params (sampling_func, trg_vocab)
# TODO: we may be able to make this function faster by passing multiple sources for sampling at the same damn time
# TODO: or by avoiding the for loop somehow
def sampling_func(source_seq, num_samples=1):

    def _get_true_length(seqs, vocab):
        try:
            lens = []
            for r in seqs.tolist():
                lens.append(r.index(vocab['</S>']) + 1)
            return lens
        except ValueError:
            return [seqs.shape[1] for _ in range(seqs.shape[0])]

    # samples = []
    # for _ in range(num_samples):
        # outputs of self.sampling_fn = outputs of sequence_generator.generate: next_states + [next_outputs] +
        #                 list(next_glimpses.values()) + [next_costs])
        # _1, outputs, _2, _3, costs = theano_sample_func(source_seq[None, :])
        # if we are generating a single sample, the length of the output will be len(source_seq)*2
        # see decoder.generate
        # the output is a [seq_len, 1] array
        # outputs = outputs.reshape(outputs.shape[0])
        # outputs = outputs[:_get_true_length(outputs, trg_vocab)]
        # samples.append(outputs)

    inputs = numpy.tile(source_seq[None, :], (num_samples, 1))
    # the output is [seq_len, batch]
    _1, outputs, _2, _3, costs = theano_sample_func(inputs)
    outputs = outputs.T

    # TODO: this step could be avoided by computing the samples mask in a different way
    lens = _get_true_length(outputs, trg_vocab)
    samples = [s[:l] for s,l in zip(outputs.tolist(), lens)]

    return samples


src_stream = get_textfile_stream(source_file=exp_config['src_data'], src_vocab=exp_config['src_vocab'],
                                         src_vocab_size=exp_config['src_vocab_size'])

# test_source_stream.sources = ('sources',)
trg_stream = get_textfile_stream(source_file=exp_config['trg_data'], src_vocab=exp_config['trg_vocab'],
                                         src_vocab_size=exp_config['trg_vocab_size'])

# Merge them to get a source, target pair
training_stream = Merge([src_stream,
                         trg_stream],
                         ('source', 'target'))

# Filter sequences that are too long
training_stream = Filter(training_stream,
                         predicate=_too_long(seq_len=exp_config['seq_len']))

# sampling_transformer = MTSampleStreamTransformer(sampling_func, fake_score, num_samples=5)
sampling_transformer = MTSampleStreamTransformer(sampling_func, sentence_level_bleu, num_samples=exp_config['n_samples'])

training_stream = Mapping(training_stream, sampling_transformer, add_sources=('samples', 'scores'))


class FlattenSamples(Transformer):
    """Adds padding to variable-length sequences.

    When your batches consist of variable-length sequences, use this class
    to equalize lengths by adding zero-padding. To distinguish between
    data and padding masks can be produced. For each data source that is
    masked, a new source will be added. This source will have the name of
    the original source with the suffix ``_mask`` (e.g. ``features_mask``).

    Elements of incoming batches will be treated as numpy arrays (i.e.
    using `numpy.asarray`). If they have more than one dimension,
    all dimensions except length, that is the first one, must be equal.

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
#         sources = []
#         for source in self.data_stream.sources:
#             sources.append(source)
#             if source in self.mask_sources:
#                 sources.append(source + '_mask')
#         return tuple(sources)

    def transform_batch(self, batch):
        batch_with_flattened_samples = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
#             if source not in self.mask_sources:
#                 batch_with_masks.append(source_batch)
#                 continue
            if source == 'samples':
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
#                 copy each source seqoyuence self.n_samples times, but keep the tensor 2d

                expanded_source = []
                for ins in source_batch:
                    expanded_source.extend([ins for _ in range(self.n_samples)])

                batch_with_expanded_source.append(expanded_source)
            else:
                batch_with_expanded_source.append(source_batch)

        return tuple(batch_with_expanded_source)



# Replace out of vocabulary tokens with unk token
# training_stream = Mapping(training_stream,
#                  _oov_to_unk(src_vocab_size=exp_config['src_vocab_size'],
#                              trg_vocab_size=exp_config['trg_vocab_size'],
#                              unk_id=exp_config['unk_id']))

# Build a batched version of stream to read k batches ahead
training_stream = Batch(training_stream,
               iteration_scheme=ConstantScheme(
                   exp_config['batch_size']*exp_config['sort_k_batches']))

# Sort all samples in the read-ahead batch
training_stream = Mapping(training_stream, SortMapping(_length))

# Convert it into a stream again
training_stream = Unpack(training_stream)

# Construct batches from the stream with specified batch size
training_stream = Batch(
    training_stream, iteration_scheme=ConstantScheme(exp_config['batch_size']))

# Pad sequences that are short
# IDEA: add a transformer which flattens the target samples before we add the mask
flat_sample_stream = FlattenSamples(training_stream)

expanded_source_stream = CopySourceNTimes(flat_sample_stream, n_samples=exp_config['n_samples'])

# TODO: some sources can be excluded from the padding Op, but since blocks matches sources with input variable
# TODO: names, it's not critical
masked_stream = PaddingWithEOS(
    expanded_source_stream, [exp_config['src_vocab_size'] - 1, exp_config['trg_vocab_size'] - 1])


def create_model(encoder, decoder):

    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')

#     target_samples = tensor.tensor3('samples').astype('int64')
#     target_samples_mask = tensor.tensor3('target_samples_mask').astype('int64')
    samples = tensor.lmatrix('samples')
    samples_mask = tensor.matrix('samples_mask')

    # scores is (batch, samples)
    scores = tensor.matrix('scores')
    # We don't need a scores mask because there should be the same number of scores for each instance
    # num samples is a hyperparameter of the model

    # the name is important to make sure pre-trained params get loaded correctly
#     decoder.name = 'decoder'

    # This is the part that is different for the MinimumRiskSequenceGenerator
    cost = decoder.expected_cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, samples, samples_mask, scores)


    return cost


def main(model, cost, config, tr_stream, dev_stream, use_bokeh=False):

    # Set the parameters from a trained models (.npz file)
    logger.info("Loading parameters from model: {}".format(exp_config['saved_parameters']))
    # Note the brick delimeter='-' is here for legacy reasons because blocks changed the serialization API
    param_values = LoadNMT.load_parameter_values(config['saved_parameters'], brick_delimiter='-')
    LoadNMT.set_model_parameters(model, param_values)

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # create the training directory, and copy this config there if directory doesn't exist
#     if not os.path.isdir(config['saveto']):
#         os.makedirs(config['saveto'])
#         shutil.copy(config['config_file'], config['saveto'])

    # Set extensions
    logger.info("Initializing extensions")
    extensions = [
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
         CheckpointNMT(config['saveto'],
                       every_n_batches=config['save_freq'])
    ]


    # Set up beam search and sampling computation graphs if necessary

    if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
        logger.info("Building sampling model")
        sampling_representation = train_encoder.apply(
            theano_sampling_input, tensor.ones(theano_sampling_input.shape))
        # TODO: the generated output actually contains several more values, ipdb to see what they are
        generated = train_decoder.generate(theano_sampling_input, sampling_representation)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[train_decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs

    # Add sampling
    if config['hook_samples'] >= 1:
        logger.info("Building sampler")
        extensions.append(
            Sampler(model=search_model, data_stream=tr_stream,
                    hook_samples=config['hook_samples'],
                    every_n_batches=config['sampling_freq'],
                    src_vocab_size=config['src_vocab_size']))

    # Add early stopping based on bleu
    if config['bleu_script'] is not None:
        logger.info("Building bleu validator")
        extensions.append(
            BleuValidator(theano_sampling_input, samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          normalize=config['normalized_bleu'],
                          every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot(config['model_save_directory'], channels=[['decoder_cost_cost'], ['validation_set_bleu_score']],
                 every_n_batches=10))

    # Set up training algorithm
    logger.info("Initializing training algorithm")
    # if there is dropout or random noise, we need to use the output of the modified graph
#     if config['dropout'] < 1.0 or config['weight_noise_ff'] > 0.0:
#         algorithm = GradientDescent(
#             cost=cg.outputs[0], parameters=cg.parameters,
#             step_rule=CompositeRule([StepClipping(config['step_clipping']),
#                                      eval(config['step_rule'])()])
#         )
#     else:
#         algorithm = GradientDescent(
#             cost=cost, parameters=cg.parameters,
#             step_rule=CompositeRule([StepClipping(config['step_clipping']),
#                                      eval(config['step_rule'])()])
#         )

    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                 eval(config['step_rule'])()],
                               ),
        on_unused_sources='warn'
    )

    # enrich the logged information
    extensions.append(
        Timing(every_n_batches=100)
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()


training_cost = create_model(train_encoder, train_decoder)


# Set up training model
logger.info("Building model")
train_model = Model(training_cost)


# test_iter = masked_stream.get_epoch_iterator()

# source, source_mask, target, target_mask, samples, samples_mask, scores, scores_mask = test_iter.next()

# train_model.inputs

# test_func = train_model.get_theano_function()

# scores = scores.astype('float32')
# out = test_func(scores, samples_mask, source_mask, source, samples)


# numpy.exp(out)
# out[0].shape
# out
# scores
# src_ivocab = {v:k for k,v in src_vocab.items()}

dev_stream = get_dev_stream(**exp_config)

main(train_model, training_cost, exp_config, masked_stream, dev_stream=dev_stream, use_bokeh=True)


