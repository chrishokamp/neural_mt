from __future__ import print_function
import logging

import os
import shutil
from collections import Counter
from theano import tensor
from toolz import merge
import numpy
import pickle
import codecs
import re
import time
import json
from subprocess import Popen, PIPE, check_output
from collections import defaultdict

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

from machine_translation.checkpoint import CheckpointNMT, LoadNMT, RunExternalValidation
from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.sampling import BleuValidator, Sampler, SamplingBase, MeteorValidator
from machine_translation.stream import (get_tr_stream, get_dev_stream,
                                        _ensure_special_tokens)

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # GRAPH TRANSFORMATIONS FOR BETTER TRAINING

    # TODO: allow user to remove some params from the graph, for example if embeddings should be kept static
    if config.get('l2_regularization', False) is True:
        l2_reg_alpha = config['l2_regularization_alpha']
        logger.info('Applying l2 regularization with alpha={}'.format(l2_reg_alpha))
        model_weights = VariableFilter(roles=[WEIGHT])(cg.variables)

        for W in model_weights:
            cost = cost + (l2_reg_alpha * (W ** 2).sum())

        # why do we need to name the cost variable? Where did the original name come from?
        cost.name = 'decoder_cost_cost'

    cg = ComputationGraph(cost)

    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        # this is the probability of dropping out, so you probably want to make it <=0.5
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

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

    # create the training directory, and copy this config there if directory doesn't exist
    if not os.path.isdir(config['saveto']):
        os.makedirs(config['saveto'])
        shutil.copy(config['config_file'], config['saveto'])

    # Set extensions
    logger.info("Initializing extensions")
    extensions = []

    # Set up beam search and sampling computation graphs if necessary
    if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
        logger.info("Building sampling model")
        sampling_representation = encoder.apply(
            sampling_input, tensor.ones(sampling_input.shape))
        # note that generated containes several different outputs
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

    # WORKING: remove these validators in favor of Async
    # TODO: implement burn-in in the validation extension (don't fire until we're past the burn-in iteration)
    # Add early stopping based on bleu
    # if config.get('bleu_script', None) is not None:
    #     logger.info("Building bleu validator")
    #     extensions.append(
    #         BleuValidator(sampling_input, samples=samples, config=config,
    #                       model=search_model, data_stream=dev_stream,
    #                       normalize=config['normalized_bleu'],
    #                       every_n_batches=config['bleu_val_freq']))

    # Add early stopping based on Meteor
    # if config.get('meteor_directory', None) is not None:
    #     logger.info("Building meteor validator")
    #     extensions.append(
    #         MeteorValidator(sampling_input, samples=samples, config=config,
    #                       model=search_model, data_stream=dev_stream,
    #                       normalize=config['normalized_bleu'],
    #                       every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    # if config['reload']:
    #     extensions.append(LoadNMT(config['saveto']))


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
    extensions.extend(
        [Timing(every_n_batches=100),
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
        CheckpointNMT(config['saveto'],
                      every_n_batches=config['save_freq'])]
    )

    extensions.append(
        RunExternalValidation(config=config, every_n_batches=config['bleu_val_freq'])
    )

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot(config['model_save_directory'], channels=[['decoder_cost_cost'], ['validation_set_bleu_score'],
                                                           ['validation_set_meteor_score']],
                 every_n_batches=1))

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


# TODO: use this function to set up training as well -- see IMT repo
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
    logger.info("Loading parameters from model: {}".format(exp_config['saved_parameters']))

    # load the parameter values from an .npz file if the `saved_parameters` field is present in the config
    param_values = LoadNMT.load_parameter_values(exp_config['saved_parameters'], brick_delimiter=exp_config.get('brick_delimiter', None))
    LoadNMT.set_model_parameters(model, param_values)

    return beam_search, sampling_input


class NMTPredictor:
    """"Uses a trained NMT model to do prediction"""

    sutils = SamplingBase()

    def __init__(self, exp_config):

        self.beam_search, self.sampling_input = load_params_and_get_beam_search(exp_config)

        self.exp_config = exp_config
        # how many hyps should be output (only used in file prediction mode)
        self.n_best = exp_config.get('n_best', 1)

        self.source_lang = exp_config.get('source_lang', 'en')
        self.target_lang = exp_config.get('target_lang', 'es')

        tokenize_script = exp_config.get('tokenize_script', None)
        detokenize_script = exp_config.get('detokenize_script', None)
        if tokenize_script is not None and detokenize_script is not None:
            self.tokenizer_cmd = [tokenize_script, '-l', self.source_lang, '-q', '-', '-no-escape', '1']
            self.detokenizer_cmd = [detokenize_script, '-l', self.target_lang, '-q', '-']
        else:
            self.tokenizer_cmd = None
            self.detokenizer_cmd = None

        # this index will get overwritten with the EOS token by _ensure_special_tokens
        # IMPORTANT: the index must be created in the same way it was for training,
        # otherwise the predicted indices will be nonsense
        # Make sure that src_vocab_size and trg_vocab_size are correct in your configuration
        self.src_eos_idx = exp_config['src_vocab_size'] - 1
        self.trg_eos_idx = exp_config['trg_vocab_size'] - 1

        self.unk_idx = exp_config['unk_id']

        # Get vocabularies and inverse indices
        # Note: _ensure_special_tokens will _overwrite_ anything at bos_idx, src_idx, and unk_idx
        self.src_vocab = _ensure_special_tokens(
            pickle.load(open(exp_config['src_vocab'])), bos_idx=0,
            eos_idx=self.src_eos_idx, unk_idx=self.unk_idx)
        self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        self.trg_vocab = _ensure_special_tokens(
            pickle.load(open(exp_config['trg_vocab'])), bos_idx=0,
            eos_idx=self.trg_eos_idx, unk_idx=self.unk_idx)
        self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}

        self.unk_idx = self.unk_idx

    def map_idx_or_unk(self, sentence, index, unknown_token='<UNK>'):
        if type(sentence) is str or type(sentence) is unicode:
            sentence = sentence.split()
        return [index.get(w, unknown_token) for w in sentence]

    def predict_file(self, input_file, output_file=None):
        tokenize = self.tokenizer_cmd is not None
        detokenize = self.detokenizer_cmd is not None

        if output_file is not None:
            ftrans = codecs.open(output_file, 'wb', encoding='utf8')
        else:
            # cut off the language suffix to make output file name
            output_file = '.'.join(input_file.split('.')[:-1]) + '.trans.out'
            ftrans = codecs.open(output_file, 'wb', encoding='utf8')

        logger.info("Started translation, will output {} translations for each segment"
                    .format(self.n_best))
        total_cost = 0.0

        # TODO: the tokenizer throws an error when the input file is opened with encoding='utf8'
        # why would this happen?
        with codecs.open(input_file, encoding='utf8') as inp:
            for i, line in enumerate(inp.read().strip().split('\n')):
                logger.info("Translating segment: {}".format(i))

                translations, costs = self.predict_segment(line, n_best=self.n_best,
                                                           tokenize=tokenize, detokenize=detokenize)

                # predict_segment returns a list of hyps, we just take the best one
                nbest_translations = translations[:self.n_best]
                nbest_costs = costs[:self.n_best]

                if self.n_best == 1:
                    ftrans.write((nbest_translations[0] + '\n').decode('utf8'))
                    total_cost += nbest_costs[0]
                else:
                    # one blank line to separate each nbest list
                    ftrans.write('\n'.join(nbest_translations).decode('utf8') + '\n\n')
                    total_cost += sum(nbest_costs)

                if i != 0 and i % 100 == 0:
                    logger.info("Translated {} lines of test set...".format(i))

        logger.info("Saved translated output to: {}".format(ftrans.name))
        logger.info("Total cost of the test: {}".format(total_cost))
        ftrans.close()

        return output_file

    def predict_segment(self, segment, n_best=1, tokenize=False, detokenize=False):
        """
        Do prediction for a single segment, which is a list of token idxs

        Parameters
        ----------
        segment: list[int] : a list of int indexes representing the input sequence in the source language
        n_best: int : how many hypotheses to return (must be <= beam_size)
        tokenize: bool : does the source segment need to be tokenized first?
        detokenize: bool : do the output hypotheses need to be detokenized?

        Returns
        -------
        trans_out: str : the best translation according to beam search
        cost: float : the cost of the best translation

        """


        if tokenize:
            tokenizer = Popen(self.tokenizer_cmd, stdin=PIPE, stdout=PIPE)
            segment, _ = tokenizer.communicate(segment)

        segment = self.map_idx_or_unk(segment, self.src_vocab, self.unk_idx)
        segment += [self.src_eos_idx]

        seq = NMTPredictor.sutils._oov_to_unk(
            segment, self.exp_config['src_vocab_size'], self.unk_idx)
        input_ = numpy.tile(seq, (self.exp_config['beam_size'], 1))

        # draw sample, checking to ensure we don't get an empty string back
        trans, costs = \
            self.beam_search.search(
                input_values={self.sampling_input: input_},
                max_length=3*len(seq), eol_symbol=self.trg_eos_idx,
                ignore_first_eol=True)

        # normalize costs according to the sequence lengths
        if self.exp_config['normalized_bleu']:
            lengths = numpy.array([len(s) for s in trans])
            costs = costs / lengths

        best_n_hyps = []
        best_n_costs = []
        best_n_idxs = numpy.argsort(costs)[:n_best]
        for j, idx in enumerate(best_n_idxs):
            try:
                trans_out = trans[idx]
                cost = costs[idx]

                # convert idx to words
                # `line` is a tuple with one item
                try:
                    assert trans_out[-1] == self.trg_eos_idx, 'Target hypothesis should end with the EOS symbol'
                    trans_out = trans_out[:-1]
                    src_in = NMTPredictor.sutils._idx_to_word(segment, self.src_ivocab)
                    trans_out = NMTPredictor.sutils._idx_to_word(trans_out, self.trg_ivocab)
                except AssertionError as e:
                    src_in = NMTPredictor.sutils._idx_to_word(segment, self.src_ivocab)
                    trans_out = NMTPredictor.sutils._idx_to_word(trans_out, self.trg_ivocab)
                    logger.error("ERROR: {} does not end with the EOS symbol".format(trans_out))
                    logger.error("I'm continuing anyway...")
            # TODO: why would this error happen?
            except ValueError:
                logger.info("Can NOT find a translation for line: {}".format(src_in))
                trans_out = '<UNK>'
                cost = 0.

            if detokenize:
                detokenizer = Popen(self.detokenizer_cmd, stdin=PIPE, stdout=PIPE)
                trans_out, _ = detokenizer.communicate(trans_out)
                # strip off the eol symbol
                trans_out = trans_out.strip()

            # TODO: remove this quick hack
            trans_out = trans_out.replace('<UNK>', 'UNK')

            logger.info("Source: {}".format(src_in))
            logger.info("Target Hypothesis: {}".format(trans_out))

            best_n_hyps.append(trans_out)
            best_n_costs.append(cost)

        return best_n_hyps, best_n_costs


def run(mode, config_obj, bokeh):
    if mode == 'train':
        # Get data streams and call main
        main(config_obj, get_tr_stream(**config_obj),
             get_dev_stream(**config_obj), bokeh)
    elif mode == 'predict':
        predictor = NMTPredictor(config_obj)
        predictor.predict_file(config_obj['test_set'], config_obj.get('translated_output_file', None))

    # TODO: let user configure which evaluation metrics to use
    elif mode == 'evaluate':
        logger.info("Started Evaluation: ")
        model_name = config_obj.get('model_name', 'default_model')
        # TODO: we need a way to keep track of the evaluations from all models, but they are running async
        evaluation_report_path = os.path.join(config_obj['saveto'], 'evaluation_reports')

        # load existing evaluation info if this model has already been evaluated
        if not os.path.isdir(evaluation_report_path):
            os.makedirs(evaluation_report_path)

        val_start_time = time.time()

        evaluation_report = []

        # translate if necessary, write output file, call external evaluation tools and show output
        translated_output_file = config_obj.get('translated_output_file', None)
        if translated_output_file is not None and os.path.isfile(translated_output_file):
            logger.info('{} already exists, so I\'m evaluating the BLEU score of this file with respect to the ' +
                        'reference that you provided: {}'.format(translated_output_file,
                                                                 config_obj['test_gold_refs']))
        else:
            predictor = NMTPredictor(config_obj)
            logger.info('Translating: {}'.format(config_obj['test_set']))
            translated_output_file = predictor.predict_file(config_obj['test_set'],
                                                            translated_output_file)
            logger.info('Translated: {}, output was written to: {}'.format(config_obj['test_set'],
                                                                           translated_output_file))

        # If this is a subword system, and user asked for normalization, do it
        if config_obj.get('normalize_subwords', False):
            with codecs.open(translated_output_file, encoding='utf8') as output:
                lines = output.readlines()
            with codecs.open(translated_output_file, 'w', encoding='utf8') as output:
                for line in lines:
                    # sed "s/@@ //g"
                    output.write(re.sub(r'@@ ', '', line))

        # if user wants BOS and/or EOS tokens cut off, do it
        if config_obj.get('remove_bos', False):
            with codecs.open(translated_output_file, encoding='utf8') as output:
                lines = output.readlines()
            with codecs.open(translated_output_file, 'w', encoding='utf8') as output:
                for line in lines:
                    output.write(re.sub(r'^' + config_obj['bos_token'] + ' ', '', line))
        if config_obj.get('remove_eos', False):
            with codecs.open(translated_output_file, encoding='utf8') as output:
                lines = output.readlines()
            with codecs.open(translated_output_file, 'w', encoding='utf8') as output:
                for line in lines:
                    output.write(re.sub(config_obj['eos_token'], '', line))

        # BLEU
        # get gold refs
        multibleu_cmd = ['perl', config_obj['bleu_script'],
                         config_obj['test_gold_refs'], '<']

        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE)

        with codecs.open(translated_output_file, encoding='utf8') as hyps:
            for l in hyps.read().strip().split('\n'):
                # send the line to the BLEU script
                print(l.encode('utf8'), file= mb_subprocess.stdin)
                mb_subprocess.stdin.flush()

            # send end of file, read output.
            mb_subprocess.stdin.close()
            stdout = mb_subprocess.stdout.readline()
            logger.info(stdout)
            out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
            logger.info("Validation Took: {} minutes".format(float(time.time() - val_start_time) / 60.))
            assert out_parse is not None

            # extract the score
            bleu_score = float(out_parse.group()[6:])
            logger.info('BLEU SCORE: {}'.format(bleu_score))
            mb_subprocess.terminate()
            evaluation_report.append(u'{} {} {}'.format('bleu', bleu_score, model_name))

        # Meteor
        meteor_directory = config_obj.get('meteor_directory', None)
        if meteor_directory is not None:
            target_language = config_obj.get('target_lang', 'de')
            # java -Xmx2G -jar meteor-*.jar test reference - l en - norm
            # Note: not using the `-norm` parameter with METEOR since the references are already tokenized
            meteor_cmd = ['java', '-Xmx4G', '-jar', os.path.join(meteor_directory, 'meteor-1.5.jar'),
                          translated_output_file, config_obj['test_gold_refs'], '-l', target_language, '-norm']

            meteor_output = check_output(meteor_cmd)
            meteor_score = float(meteor_output.strip().split('\n')[-1].split()[-1])
            logger.info('METEOR SCORE: {}'.format(meteor_score))
            evaluation_report.append(u'{} {} {}'.format('meteor', bleu_score, model_name))

        # touch a file for each row in evaluation_report, the file name is the result
        for l in evaluation_report:
            open(os.path.join(evaluation_report_path, l), 'w').close()
        logger.info('Wrote evaluation report files to: {}'.format(evaluation_report_path))

    elif mode == 'server':

        import sys
        sys.path.append('.')
        from server import run_nmt_server

        # start restful server and log its port
        predictor = NMTPredictor(config_obj)
        run_nmt_server(predictor)

    else:
        print('ERROR: mode unknown: {}'.format(mode))

