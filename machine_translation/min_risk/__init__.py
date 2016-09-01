"""
This module implements the main method for minimum risk training

"""

import logging

import os
import shutil
from collections import Counter
from theano import tensor
from toolz import merge
import numpy
import pickle
from subprocess import Popen, PIPE
import codecs

from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)
from fuel.schemes import ConstantScheme

from blocks.algorithms import (GradientDescent, StepClipping,
                               CompositeRule, Adam, AdaDelta)
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import WEIGHT
from blocks.select import Selector
from blocks.search import BeamSearch
from blocks_extras.extensions.plot import Plot

from machine_translation.checkpoint import CheckpointNMT, LoadNMT
from machine_translation.model import BidirectionalEncoder

from machine_translation.evaluation import sentence_level_bleu, sentence_level_meteor
from machine_translation.stream import (get_textfile_stream, _too_long, _oov_to_unk,
                                        ShuffleBatchTransformer, PaddingWithEOS, FlattenSamples)

from nn_imt.sample import BleuValidator, IMT_F1_Validator, Sampler, SamplingBase
from nn_imt.model import NMTPrefixDecoder
from nn_imt.sample import SampleFunc

from nn_imt.stream import (PrefixSuffixStreamTransformer, CopySourceAndTargetToMatchPrefixes,
                           IMTSampleStreamTransformer, CopySourceAndPrefixNTimes, _length,
			               filter_by_sample_score)
from nn_imt.evaluation import sentence_level_imt_f1

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# WORKING: searching for NaN cause
#import theano
#theano.config.optimizer = 'None'
#theano.config.traceback.limit = 30


# TODO: where do the variables of this model get set?
# Note: this is to create the sampling model -- the cost model gets built from the output of this function
def get_sampling_model_and_input(exp_config):
    # Create Theano variables
    encoder = BidirectionalEncoder(
        exp_config['src_vocab_size'], exp_config['enc_embed'], exp_config['enc_nhids'])

    # Note: the 'min_risk' kwarg tells the decoder which sequence_generator and cost_function to use
    decoder = NMTPrefixDecoder(
        exp_config['trg_vocab_size'], exp_config['dec_embed'], exp_config['dec_nhids'],
        exp_config['enc_nhids'] * 2, loss_function='min_risk')

    # rename to match baseline NMT systems
    decoder.name = 'decoder'

    # Create Theano variables
    logger.info('Creating theano variables')
    sampling_source_input = tensor.lmatrix('source')
    sampling_prefix_input = tensor.lmatrix('prefix')

    # Get generation model
    logger.info("Building sampling model")

    sampling_source_representation = encoder.apply(
        sampling_source_input, tensor.ones(sampling_source_input.shape))

    # WORKING: get the costs (logprobs) from here
    # WORKING: make a theano variable for the costs, pass it to expected_cost
    # WORKING: how _exactly_ are the costs computed?
    # return (next_states + [next_outputs] +
    #         list(next_glimpses.values()) + [next_costs])
    generated = decoder.generate(sampling_source_input,
                                 sampling_source_representation,
                                 target_prefix=sampling_prefix_input)

    # build the model that will let us get a theano function from the sampling graph
    logger.info("Creating Sampling Model...")
    sampling_model = Model(generated)

    # TODO: update clients with sampling_context_input
    return sampling_model, sampling_source_input, sampling_prefix_input, encoder, decoder, generated


def setup_model_and_stream(exp_config, source_vocab, target_vocab):

    # TODO: this line is a mess
    sample_model, theano_sampling_source_input, theano_sampling_context_input, train_encoder, \
    train_decoder, generated = \
        get_sampling_model_and_input(exp_config)

    trg_vocab = target_vocab
    trg_vocab_size = exp_config['trg_vocab_size']
    src_vocab = source_vocab
    src_vocab_size = exp_config['src_vocab_size']

    theano_sample_func = sample_model.get_theano_function()
    sampling_func = SampleFunc(theano_sample_func, trg_vocab)

    # TODO: move stream creation to nn_imt.stream
# def get_textfile_stream(source_file=None, src_vocab=None, src_vocab_size=30000,
#                         unk_id=1, bos_token=None):
    src_stream = get_textfile_stream(source_file=exp_config['src_data'], src_vocab=exp_config['src_vocab'],
                                     src_vocab_size=exp_config['src_vocab_size'], unk_id=exp_config['unk_id'],
                                     bos_token='<S>')

    trg_stream = get_textfile_stream(source_file=exp_config['trg_data'], src_vocab=exp_config['trg_vocab'],
                                     src_vocab_size=exp_config['trg_vocab_size'], unk_id=exp_config['unk_id'],
                                     bos_token='<S>')

    # text file stream
    training_stream = Merge([src_stream,
                             trg_stream],
                             ('source', 'target'))

    # Filter sequences that are too long (Note this may break)
    training_stream = Filter(training_stream,
                    predicate=_too_long(seq_len=exp_config['seq_len']))


    # Replace out of vocabulary tokens with unk token
    # TODO: doesn't the TextFile stream do this anyway?
    training_stream = Mapping(training_stream,
                              _oov_to_unk(src_vocab_size=exp_config['src_vocab_size'],
                                          trg_vocab_size=exp_config['trg_vocab_size'],
                                          unk_id=exp_config['unk_id']))

    # add in the prefix and suffix seqs
    # working: add the sample ratio
    logger.info('Sample ratio is: {}'.format(exp_config.get('sample_ratio', 1.)))
    training_stream = Mapping(training_stream,
                              PrefixSuffixStreamTransformer(sample_ratio=exp_config.get('sample_ratio', 1.)),
                                                            add_sources=('target_prefix', 'target_suffix'))

    training_stream = Mapping(training_stream, CopySourceAndTargetToMatchPrefixes(training_stream))

    # changing stream.produces_examples is a little hack which lets us use Unpack to flatten
    training_stream.produces_examples = False

    # flatten the stream back out into (source, target, target_prefix, target_suffix)
    training_stream = Unpack(training_stream)

    # METEOR
    trg_ivocab = {v:k for k,v in trg_vocab.items()}

    # TODO: Implement smoothed BLEU
    # TODO: Implement first-word accuracy (bilingual language model)

    min_risk_score_func = exp_config.get('min_risk_score_func', 'bleu')

    if min_risk_score_func == 'meteor':
        sampling_transformer = IMTSampleStreamTransformer(sampling_func,
                                                          sentence_level_meteor,
                                                          num_samples=exp_config['n_samples'],
                                                          trg_ivocab=trg_ivocab,
                                                          lang=exp_config['target_lang'],
                                                          meteor_directory=exp_config['meteor_directory']
                                                         )
    elif min_risk_score_func == 'imt_f1':
        sampling_transformer = IMTSampleStreamTransformer(sampling_func,
                                                          sentence_level_imt_f1,
                                                          num_samples=exp_config['n_samples'])
    # BLEU is default
    else:
        sampling_transformer = IMTSampleStreamTransformer(sampling_func,
                                                          sentence_level_bleu,
                                                          num_samples=exp_config['n_samples'])

    training_stream = Mapping(training_stream, sampling_transformer, add_sources=('samples', 'seq_probs', 'scores'))

    # now filter out segments whose samples are too good or too bad
    training_stream = Filter(training_stream,
                             predicate=filter_by_sample_score)

    # Now make a very big batch that we can shuffle
    # Build a batched version of stream to read k batches ahead
    shuffle_batch_size = exp_config['shuffle_batch_size']
    training_stream = Batch(training_stream,
                            iteration_scheme=ConstantScheme(shuffle_batch_size))

    training_stream = ShuffleBatchTransformer(training_stream)

    # unpack it again
    training_stream = Unpack(training_stream)

    # Build a batched version of stream to read k batches ahead
    batch_size = exp_config['batch_size']
    sort_k_batches = exp_config['sort_k_batches']
    training_stream = Batch(training_stream,
                            iteration_scheme=ConstantScheme(batch_size * sort_k_batches))

    # Sort all samples in the read-ahead batch
    training_stream = Mapping(training_stream, SortMapping(_length))

    # Convert it into a stream again
    training_stream = Unpack(training_stream)

    # Construct batches from the stream with specified batch size
    training_stream = Batch(training_stream, iteration_scheme=ConstantScheme(batch_size))

    # IDEA: add a transformer which flattens the target samples before we add the mask
    flat_sample_stream = FlattenSamples(training_stream)

    expanded_source_stream = CopySourceAndPrefixNTimes(flat_sample_stream, n_samples=exp_config['n_samples'])

    # Pad sequences that are short
    # TODO: is it correct to blindly pad the target_prefix and the target_suffix?
    # Note: we shouldn't need to pad the seq_probs because there is only one per sequence
    # TODO: DEVELOPMENT HACK
    exp_config['suffix_length'] = 1
    exp_config['truncate_sources'] = ['target_suffix']
    configurable_padding_args = {
        'suffix_length': exp_config.get('suffix_length', None),
        'truncate_sources': exp_config.get('truncate_sources', [])
    }
    import ipdb; ipdb.set_trace()
    masked_stream = PaddingWithEOS(
        expanded_source_stream, [src_vocab_size - 1, trg_vocab_size - 1, trg_vocab_size - 1, trg_vocab_size - 1, trg_vocab_size - 1],
        mask_sources=('source', 'target', 'target_prefix', 'target_suffix', 'samples'), **configurable_padding_args
    )

    return train_encoder, train_decoder, theano_sampling_source_input, theano_sampling_context_input, generated, masked_stream


# create the model for training
# TODO: implement the expected_cost multimodal decoder
def create_model(encoder, decoder, smoothing_constant=0.005):

    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')

    samples = tensor.lmatrix('samples')
    samples_mask = tensor.matrix('samples_mask')

    target_prefix = tensor.lmatrix('target_prefix')
    prefix_mask = tensor.matrix('target_prefix_mask')

    # scores is (batch, samples)
    scores = tensor.matrix('scores')
    # We don't need a scores mask because there should be the same number of scores for each instance
    # num samples is a hyperparameter of the model

    # This is the part that is different for the MinimumRiskSequenceGenerator
    cost = decoder.expected_cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, samples, samples_mask, scores,
        prefix_outputs=target_prefix,
        prefix_mask=prefix_mask,
        smoothing_constant=smoothing_constant
    )

    return cost


# def main(model, cost, config, tr_stream, dev_stream, use_bokeh=False):
def main(exp_config, source_vocab, target_vocab, dev_stream, use_bokeh=True):


# def setup_model_and_stream(exp_config, source_vocab, target_vocab):
    # def setup_model_and_stream(exp_config, source_vocab, target_vocab):
    train_encoder, train_decoder, theano_sampling_source_input, theano_sampling_context_input, generated, masked_stream = setup_model_and_stream(exp_config, source_vocab, target_vocab)
    cost = create_model(train_encoder, train_decoder, exp_config.get('imt_smoothing_constant', 0.005))

# Set up training model
    logger.info("Building model")
    train_model = Model(cost)

    # Set the parameters from a trained models (.npz file)
    logger.info("Loading parameters from model: {}".format(exp_config['saved_parameters']))
    # Note the brick delimeter='-' is here for legacy reasons because blocks changed the serialization API
    param_values = LoadNMT.load_parameter_values(exp_config['saved_parameters'], brick_delimiter=exp_config.get('brick_delimiter', None))
    LoadNMT.set_model_parameters(train_model, param_values)

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # GRAPH TRANSFORMATIONS FOR BETTER TRAINING
    if exp_config.get('l2_regularization', False) is True:
        l2_reg_alpha = exp_config['l2_regularization_alpha']
        logger.info('Applying l2 regularization with alpha={}'.format(l2_reg_alpha))
        model_weights = VariableFilter(roles=[WEIGHT])(cg.variables)

        for W in model_weights:
            cost = cost + (l2_reg_alpha * (W ** 2).sum())

        # why do we need to rename the cost variable? Where did the original name come from?
        cost.name = 'decoder_cost_cost'

    cg = ComputationGraph(cost)

    # apply dropout for regularization
    # Note dropout variables are hard-coded here
    if exp_config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        # this is the probability of dropping out, so you probably want to make it <=0.5
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, exp_config['dropout'])

    # create the training directory, and copy this config there if directory doesn't exist
    if not os.path.isdir(exp_config['saveto']):
        os.makedirs(exp_config['saveto'])
        # TODO: mv the actual config file once we switch to .yaml for min-risk
        shutil.copy(exp_config['config_file'], exp_config['saveto'])

    # Set extensions
    logger.info("Initializing extensions")
    extensions = [
        FinishAfter(after_n_batches=exp_config['finish_after']),
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
         CheckpointNMT(exp_config['saveto'],
                       every_n_batches=exp_config['save_freq'])
    ]

    # Set up beam search and sampling computation graphs if necessary
    # TODO: change the if statement here
    if exp_config['hook_samples'] >= 1 or exp_config['bleu_script'] is not None:
        logger.info("Building sampling model")
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
    # TODO: use multimodal meteor and BLEU validator
    # TODO: add 'validator' key to IMT config
    # Add early stopping based on bleu
    if exp_config.get('bleu_script', None) is not None:
        logger.info("Building bleu validator")
        extensions.append(
            BleuValidator(theano_sampling_source_input, theano_sampling_context_input,
                          samples=samples, config=exp_config,
                          model=search_model, data_stream=dev_stream,
                          src_vocab=source_vocab,
                          trg_vocab=target_vocab,
                          normalize=exp_config['normalized_bleu'],
                          every_n_batches=exp_config['bleu_val_freq']))

    if exp_config.get('imt_f1_validation', False) is not False:
        logger.info("Building imt F1 validator")
        extensions.append(
            IMT_F1_Validator(theano_sampling_source_input, theano_sampling_context_input,
                             samples=samples,
                             config=exp_config,
                             model=search_model, data_stream=dev_stream,
                             src_vocab=source_vocab,
                             trg_vocab=target_vocab,
                             normalize=exp_config['normalized_bleu'],
                             every_n_batches=exp_config['bleu_val_freq']))



    # Add early stopping based on Meteor
    # if exp_config.get('meteor_directory', None) is not None:
    #     logger.info("Building meteor validator")
    #     extensions.append(
    #         MeteorValidator(theano_sampling_source_input, theano_sampling_context_input,
    #                         samples=samples,
    #                         config=config,
    #                         model=search_model, data_stream=dev_stream,
    #                         src_vocab=src_vocab,
    #                         trg_vocab=trg_vocab,
    #                         normalize=config['normalized_bleu'],
    #                         every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    if exp_config['reload']:
        extensions.append(LoadNMT(exp_config['saveto']))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot(exp_config['model_save_directory'], channels=[['decoder_cost_cost', 'validation_set_imt_f1_score', 'validation_set_bleu_score', 'validation_set_meteor_score']],
                 every_n_batches=10))

    # Set up training algorithm
    logger.info("Initializing training algorithm")

    # if there is l2_regularization, dropout or random noise, we need to use the output of the modified graph
    # WORKING: try to catch and fix nan
    if exp_config['dropout'] < 1.0:
        if exp_config.get('nan_guard', False):
            from theano.compile.nanguardmode import NanGuardMode
            algorithm = GradientDescent(
                cost=cg.outputs[0], parameters=cg.parameters,
                step_rule=CompositeRule([StepClipping(exp_config['step_clipping']),
                                         eval(exp_config['step_rule'])()]),
                on_unused_sources='warn',
                theano_func_kwargs={'mode': NanGuardMode(nan_is_error=True, inf_is_error=True)}
            )
        else:
            algorithm = GradientDescent(
                cost=cg.outputs[0], parameters=cg.parameters,
                step_rule=CompositeRule([StepClipping(exp_config['step_clipping']),
                                         eval(exp_config['step_rule'])()]),
                on_unused_sources='warn'
            )
    else:
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(exp_config['step_clipping']),
                                     eval(exp_config['step_rule'])()]),
            on_unused_sources='warn'
        )

    # enrich the logged information
    extensions.append(
        Timing(every_n_batches=100)
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=train_model,
        algorithm=algorithm,
        data_stream=masked_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()


