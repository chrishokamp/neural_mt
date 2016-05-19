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
import shutil
import yaml

from fuel.datasets import Dataset
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

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
from blocks.roles import WEIGHT
from blocks.search import BeamSearch
from blocks_extras.extensions.plot import Plot

from machine_translation.checkpoint import CheckpointNMT, LoadNMT
from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.sampling import BleuValidator, Sampler, SamplingBase, MeteorValidator
from machine_translation.sample import SampleFunc
from machine_translation.stream import (_ensure_special_tokens, MTSampleStreamTransformer,
                                        get_textfile_stream, get_dev_stream, _too_long, _length, PaddingWithEOS,
                                        _oov_to_unk, CopySourceNTimes, FlattenSamples)
from machine_translation.evaluation import sentence_level_bleu, sentence_level_meteor

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

# BASEDIR = '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BERTHA-TEST_Adam_wmt-multimodal_internal_data_dropout'+\
#           '0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/'
BASEDIR = '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed'
#best_bleu_model_1455464992_BLEU31.61.npz

exp_config = {
    'src_vocab_size': 20000,
    'trg_vocab_size': 20000,
    'enc_embed': 300,
    'dec_embed': 300,
    'enc_nhids': 800,
    'dec_nhids': 800,
    'src_vocab': os.path.join(BASEDIR, 'vocab.en-de.en.pkl'),
    'trg_vocab': os.path.join(BASEDIR, 'vocab.en-de.de.pkl'),
    'src_data': os.path.join(BASEDIR, 'train.en.tok.shuf'),
    'trg_data': os.path.join(BASEDIR, 'train.de.tok.shuf'),
    'unk_id':1,
    # Bleu script that will be used (moses multi-perl in this case)
    # 'bleu_script': os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                             '../test_data/sample_experiment/tiny_demo_dataset/multi-bleu.perl'),

    # Optimization related ----------------------------------------------------
    # Batch size
    'batch_size': 4,
    # This many batches will be read ahead and sorted
    'sort_k_batches': 1,
    # Optimization step rule
    'step_rule': 'AdaDelta',
    # Gradient clipping threshold
    'step_clipping': 1.,
    # Std of weight initialization
    'weight_scale': 0.1,
    'seq_len': 40,
    # Beam-size
    'beam_size': 20,
    # dropout
    'dropout': 0.5,
    # l2_reg
    # 'l2_regularization': True,
    # 'l2_regularization_alpha': 0.001,

    # Maximum number of updates
    'finish_after': 10000,

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

    # Validation set source file
    'val_set': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/dev.en.tok',

    # Validation set gold file
    'val_set_grndtruth': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/dev.de.tok',

    # Print validation output to file
    'output_val_set': True,

    # Validation output file
    'val_set_out': '/media/1tb_drive/test_min_risk_model_save/validation_out.txt',
    'val_burn_in': 0,

    'source_lang': 'en',
    'target_lang': 'de',

    # NEW PARAM FOR MIN RISK
    'n_samples': 20,

    'meteor_directory': '/home/chris/programs/meteor-1.5',
    # 'brick_delimiter': '-'

    # 'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BERTHA-TEST_wmt-multimodal_internal_data_dropout0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/best_bleu_model_1455410311_BLEU30.38.npz',
    #'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BASELINE_WITH_REGULARIZATION0.0001_WEIGHT_SCALE_0.1_wmt-multimodal_internal_data_dropout0.5_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/best_bleu_model_1461853863_BLEU32.80.npz',
    'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BASELINE_METEOR_CHECKPOINT_WITH_REGULARIZATION0.0001_WEIGHT_SCALE_0.1_wmt-multimodal_internal_data_dropout0.5_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/best_model_1462312300_METEOR0.52.npz',
    'saveto': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/TEST-BROKEN-MIN-RISK-DROPOUT0.5-l2regNONE-n_samples50-batch20',
    'model_save_directory': 'TEST-BROKEN-MIN-RISK-DROPOUT0.5-l2regNONE-n_samples50-batch20',
}

# TODO: load config from .yaml file -- this will let us be dynamic with the run names, which is a big advantage

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

sampling_func = SampleFunc(theano_sample_func, trg_vocab)

src_stream = get_textfile_stream(source_file=exp_config['src_data'], src_vocab=exp_config['src_vocab'],
                                         src_vocab_size=exp_config['src_vocab_size'])

trg_stream = get_textfile_stream(source_file=exp_config['trg_data'], src_vocab=exp_config['trg_vocab'],
                                         src_vocab_size=exp_config['trg_vocab_size'])

# Merge them to get a source, target pair
training_stream = Merge([src_stream,
                         trg_stream],
                         ('source', 'target'))

# Filter sequences that are too long
training_stream = Filter(training_stream,
                         predicate=_too_long(seq_len=exp_config['seq_len']))

# TODO: configure min-risk score func from the yaml config

# BLEU
# sampling_transformer = MTSampleStreamTransformer(sampling_func,
#                                                  sentence_level_bleu,
#                                                  num_samples=exp_config['n_samples'])

# METEOR
trg_ivocab = {v:k for k,v in trg_vocab.items()}

# WORKING: pass kwargs through the SampleStreamTransformer to the scoring function
sampling_transformer = MTSampleStreamTransformer(sampling_func,
                                                 sentence_level_meteor,
                                                 num_samples=exp_config['n_samples'],
                                                 trg_ivocab=trg_ivocab,
                                                 lang=exp_config['target_lang'],
                                                 meteor_directory=exp_config['meteor_directory']
                                                )

training_stream = Mapping(training_stream, sampling_transformer, add_sources=('samples', 'scores'))

# Build a batched version of stream to read k batches ahead
training_stream = Batch(training_stream,
                        iteration_scheme=ConstantScheme(
                        exp_config['batch_size']*exp_config['sort_k_batches']))

# TODO: add read-ahead shuffling Mapping similar to SortMapping
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

# create the model for training
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
    param_values = LoadNMT.load_parameter_values(exp_config['saved_parameters'], brick_delimiter=exp_config.get('brick_delimiter', None))
    LoadNMT.set_model_parameters(model, param_values)

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # GRAPH TRANSFORMATIONS FOR BETTER TRAINING
    if config.get('l2_regularization', False) is True:
        l2_reg_alpha = config['l2_regularization_alpha']
        logger.info('Applying l2 regularization with alpha={}'.format(l2_reg_alpha))
        model_weights = VariableFilter(roles=[WEIGHT])(cg.variables)

        for W in model_weights:
            cost = cost + (l2_reg_alpha * (W ** 2).sum())

        # why do we need to rename the cost variable? Where did the original name come from?
        cost.name = 'decoder_cost_cost'

    cg = ComputationGraph(cost)

    # apply dropout for regularization
    # Note dropout variables are hard-coded here
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        # this is the probability of dropping out, so you probably want to make it <=0.5
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # create the training directory, and copy this config there if directory doesn't exist
    if not os.path.isdir(config['saveto']):
        os.makedirs(config['saveto'])
        # TODO: mv the actual config file once we switch to .yaml for min-risk
        # shutil.copy(config['config_file'], config['saveto'])
        # shutil.copy(config['config_file'], config['saveto'])

        with codecs.open(os.path.join(config['saveto'], 'config.yaml'), 'w', encoding='utf8') as yaml_out:
            yaml_out.write(yaml.dump(config))

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
    # TODO: change the if here
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

    # Add sampling -- TODO: sampling is broken for min-risk
    #if config['hook_samples'] >= 1:
    #    logger.info("Building sampler")
    #    extensions.append(
    #        Sampler(model=search_model, data_stream=tr_stream,
    #                hook_samples=config['hook_samples'],
    #                every_n_batches=config['sampling_freq'],
    #                src_vocab_size=config['src_vocab_size']))

    # Add early stopping based on bleu
    if config.get('bleu_script', None) is not None:
        logger.info("Building bleu validator")
        extensions.append(
            BleuValidator(theano_sampling_input, samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          normalize=config['normalized_bleu'],
                          every_n_batches=config['bleu_val_freq']))

    # Add early stopping based on bleu
    if config.get('meteor_directory', None) is not None:
        logger.info("Building meteor validator")
        extensions.append(
            MeteorValidator(theano_sampling_input, samples=samples, config=config,
                            model=search_model, data_stream=dev_stream,
                            normalize=config['normalized_bleu'],
                            every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot(config['model_save_directory'], channels=[['decoder_cost_cost'], ['validation_set_meteor_score']],
                 every_n_batches=10))

    # Set up training algorithm
    logger.info("Initializing training algorithm")

    # TODO: reenable dropout and add L2 regularization as in the main config

    # if there is l2_regularization, dropout or random noise, we need to use the output of the modified graph
    if config['dropout'] < 1.0:
        algorithm = GradientDescent(
            cost=cg.outputs[0], parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()]),
            on_unused_sources='warn'
        )
    else:
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()]),
            on_unused_sources='warn'
        )

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

dev_stream = get_dev_stream(**exp_config)

main(train_model, training_cost, exp_config, masked_stream, dev_stream=dev_stream, use_bokeh=True)


