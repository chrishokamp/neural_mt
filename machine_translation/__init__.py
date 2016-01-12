import logging

from collections import Counter
from theano import tensor
from toolz import merge
import numpy
import os
import pickle

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector
from blocks.search import BeamSearch


from machine_translation.checkpoint import CheckpointNMT, LoadNMT
from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.sampling import BleuValidator, Sampler, SamplingBase
from machine_translation.stream import (get_tr_stream, get_dev_stream,
                                        _ensure_special_tokens)

try:
    from blocks.extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)


def main(config, tr_stream, dev_stream, use_bokeh=False):

    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')
    sampling_input = tensor.lmatrix('input')

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2)
    cost = decoder.cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask)

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # Initialize model
    logger.info('Initializing model')
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
        config['weight_scale'])
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        # this is the probability of dropping out, so you probably want to make it <=0.5
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # Apply weight noise for regularization
    if config['weight_noise_ff'] > 0.0:
        logger.info('Applying weight noise to ff layers')
        enc_params = Selector(encoder.lookup).get_parameters().values()
        enc_params += Selector(encoder.fwd_fork).get_parameters().values()
        enc_params += Selector(encoder.back_fork).get_parameters().values()
        dec_params = Selector(
            decoder.sequence_generator.readout).get_parameters().values()
        dec_params += Selector(
            decoder.sequence_generator.fork).get_parameters().values()
        dec_params += Selector(decoder.transition.initial_transformer).get_parameters().values()
        cg = apply_noise(cg, enc_params+dec_params, config['weight_noise_ff'])

    # TODO: weight noise for recurrent params isn't currently implemented -- see config['weight_noise_rec']
    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
    logger.info("Total number of parameters: {}".format(len(shapes)))

    # Print parameter names
    enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                               Selector(decoder).get_parameters())
    logger.info("Parameter names: ")
    for name, value in enc_dec_param_dict.items():
        logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    logger.info("Total number of parameters: {}"
                .format(len(enc_dec_param_dict)))

    # Set up training model
    logger.info("Building model")
    training_model = Model(cost)

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
        sampling_representation = encoder.apply(
            sampling_input, tensor.ones(sampling_input.shape))
        # TODO: the generated output actually contains several more values, ipdb to see what they are
        generated = decoder.generate(sampling_input, sampling_representation)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
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
            BleuValidator(sampling_input, samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          normalize=config['normalized_bleu'],
                          every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot('Cs-En', channels=[['decoder_cost_cost']],
                 after_batch=True))

    # Set up training algorithm
    logger.info("Initializing training algorithm")
    # if there is dropout or random noise, we need to use the output of the modified graph
    if config['dropout'] < 1.0 or config['weight_noise_ff'] > 0.0:
        algorithm = GradientDescent(
            cost=cg.outputs[0], parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()])
        )
    else:
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()])
        )

    # enrich the logged information
    extensions.append(
        Timing(every_n_batches=100)
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()


# WORKING: split into functions -- get_beam_search, predict sentence,
# predict_and_save_file(beam_search, input_file, output_file=None)
def load_params_and_get_beam_search(exp_config):

    encoder = BidirectionalEncoder(
        exp_config['src_vocab_size'], exp_config['enc_embed'], exp_config['enc_nhids'])

    decoder = Decoder(
        exp_config['trg_vocab_size'], exp_config['dec_embed'], exp_config['dec_nhids'],
        exp_config['enc_nhids'] * 2)

    # Create Theano variables
    logger.info('Creating theano variables')
    sampling_input = tensor.lmatrix('source')


    # Get beam search
    logger.info("Building sampling model")
    sampling_representation = encoder.apply(
        sampling_input, tensor.ones(sampling_input.shape))
    generated = decoder.generate(sampling_input, sampling_representation)
    _, samples = VariableFilter(
        bricks=[decoder.sequence_generator], name="outputs")(
                 ComputationGraph(generated[1]))  # generated[1] is next_outputs
    beam_search = BeamSearch(samples=samples)

    # Set the parameters
    logger.info("Creating Model...")
    model = Model(generated)
    logger.info("Loading parameters from model: {}".format(exp_config['saveto']))

    # load the parameter values from an .npz file
    param_values = LoadNMT.load_parameter_values(exp_config['saved_parameters'])
    LoadNMT.set_model_parameters(model, param_values)

    return beam_search, sampling_input


def predict(exp_config):

    beam_search, sampling_input = load_params_and_get_beam_search(exp_config)

    # Get test set stream
    test_stream = get_dev_stream(
        exp_config['test_set'], exp_config['src_vocab'],
        exp_config['src_vocab_size'], exp_config['unk_id'])

    # TODO: move this to exp_config with default
    if exp_config.get('translated_output_file', None) is not None:
        ftrans = open(exp_config['translated_output_file'], 'w')
    else:
        ftrans = open(exp_config['test_set'] + '.trans.out', 'w')

    # Helper utilities
    sutils = SamplingBase()
    unk_idx = exp_config['unk_id']

    # this index will get overwritten with the EOS token by _ensure_special_tokens
    # IMPORTANT: the index must be created in the same way it was for training,
    # otherwise the predicted indices will be nonsense
    src_eos_idx = exp_config['src_vocab_size'] - 1
    trg_eos_idx = exp_config['trg_vocab_size'] - 1
    # Get target vocabulary
    trg_vocab = _ensure_special_tokens(
        pickle.load(open(exp_config['trg_vocab'])), bos_idx=0,
        eos_idx=trg_eos_idx, unk_idx=unk_idx)
    trg_ivocab = {v: k for k, v in trg_vocab.items()}

    src_vocab = _ensure_special_tokens(
        pickle.load(open(exp_config['src_vocab'])), bos_idx=0,
        eos_idx=src_eos_idx, unk_idx=unk_idx)
    src_ivocab = {v: k for k, v in src_vocab.items()}

    logger.info("Started translation: ")
    total_cost = 0.0

    # TODO: WORKING -- the model creation and prediction can be split into different functions
    # TODO: WORKING -- this is prediction for a file -- support keeping the model in memory and using as client-server
    for i, line in enumerate(test_stream.get_epoch_iterator()):
        logger.info("Translating segment: {}".format(i))
        seq = sutils._oov_to_unk(
            line[0], exp_config['src_vocab_size'], unk_idx)
        input_ = numpy.tile(seq, (exp_config['beam_size'], 1))

        # draw sample, checking to ensure we don't get an empty string back
        trans, costs = \
            beam_search.search(
                input_values={sampling_input: input_},
                max_length=3*len(seq), eol_symbol=trg_eos_idx,
                ignore_first_eol=True)

        # normalize costs according to the sequence lengths
        if exp_config['normalized_bleu']:
            lengths = numpy.array([len(s) for s in trans])
            costs = costs / lengths

        best = numpy.argsort(costs)[0]
        try:
            total_cost += costs[best]
            trans_out = trans[best]

            # convert idx to words
            # `line` is a tuple with one item
            src_in = sutils._idx_to_word(line[0], src_ivocab)
            trans_out = sutils._idx_to_word(trans_out, trg_ivocab)
        # TODO: why would this error happen?
        except ValueError:
            logger.info("Can NOT find a translation for line: {}".format(i+1))
            trans_out = '<UNK>'

        logger.info("Source: {}".format(src_in))
        logger.info("Target Hypothesis: {}".format(trans_out))


        ftrans.write(trans_out +'\n')

        if i != 0 and i % 100 == 0:
            logger.info("Translated {} lines of test set...".format(i))

    logger.info("Saved translated output to: {}".format(ftrans.name))
    logger.info("Total cost of the test: {}".format(total_cost))
    ftrans.close()
