"""Utils for checkpointing and early stopping on NMT models"""

# WORKING: extension which looks in a directory for a specific file containing validation checks, and checks i
# WORKING: anything has changed from the last time it looks
# WORKING: if something has changed, put it in the log so it can be plotted, etc...

import logging
import numpy
import os
import time
import shutil
import copy
import codecs
import yaml
import subprocess

from contextlib import closing
from six.moves import cPickle

from blocks.extensions.saveload import SAVED_TO, LOADED_FROM
from blocks.extensions import TrainingExtension, SimpleExtension
from blocks.serialization import secure_dump, load, BRICK_DELIMITER
from blocks.utils import reraise_as

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SaveLoadUtils(object):
    """Utility class for checkpointing."""

    @property
    def path_to_folder(self):
        return self.folder

    @property
    def path_to_parameters(self):
        return os.path.join(self.folder, 'params.npz')

    @property
    def path_to_iteration_state(self):
        return os.path.join(self.folder, 'iterations_state.pkl')

    @property
    def path_to_log(self):
        return os.path.join(self.folder, 'log')

    @staticmethod
    def load_parameter_values(path, brick_delimiter='|'):
        with closing(numpy.load(path)) as source:
            param_values = {}
            for name, value in source.items():
                if name != 'pkl':
                    brick_delimiter = '|'
                    # Chris: BRICK_DELIMITER is defined in blocks.serialization
                    if brick_delimiter is None:
                        name_ = name.replace(BRICK_DELIMITER, '/')
                    else:
                        name_ = name.replace(brick_delimiter, '/')
                    if not name_.startswith('/'):
                        name_ = '/' + name_
                    param_values[name_] = value
        return param_values

    @staticmethod
    def save_parameter_values(param_values, path):
        param_values = {name.replace("/", BRICK_DELIMITER): param
                        for name, param in param_values.items()}
        numpy.savez(path, **param_values)

    @staticmethod
    def set_model_parameters(model, params):
        params_this = model.get_parameter_dict()
        missing = set(params_this.keys()) - set(params.keys())
        logger.info('Note that the delimeter for parameter loading is currently hacked')
        for pname in params_this.keys():
            if pname in params:
                val = params[pname]
                if params_this[pname].get_value().shape != val.shape:
                    logger.warning(
                        " Dimension mismatch {}-{} for {}"
                        .format(params_this[pname].get_value().shape,
                                val.shape, pname))

                params_this[pname].set_value(val)
                logger.info(" Loaded to CG {:15}: {}"
                            .format(val.shape, pname))
            else:
                logger.warning(" Parameter does not exist: {}".format(pname))
        logger.info(
            " Number of parameters loaded for computation graph: {}"
            .format(len(params_this) - len(missing)))


class CheckpointNMT(SimpleExtension, SaveLoadUtils):
    """Redefines checkpointing for NMT.

        Saves only parameters (npz), iteration state (pickle) and log (pickle).

    """

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        kwargs.setdefault("after_training", True)
        super(CheckpointNMT, self).__init__(**kwargs)

    def dump_parameters(self, main_loop):
        params_to_save = main_loop.model.get_parameter_values()
        self.save_parameter_values(params_to_save,
                                   self.path_to_parameters)

    def dump_iteration_state(self, main_loop):
        secure_dump(main_loop.iteration_state, self.path_to_iteration_state)

    def dump_log(self, main_loop):
        secure_dump(main_loop.log, self.path_to_log, cPickle.dump)

    def dump(self, main_loop):
        if not os.path.exists(self.path_to_folder):
            os.mkdir(self.path_to_folder)
        print("")
        logger.info(" Saving model")
        start = time.time()
        logger.info(" ...saving parameters")
        self.dump_parameters(main_loop)
        logger.info(" ...saving iteration state")
        self.dump_iteration_state(main_loop)
        logger.info(" ...saving log")
        self.dump_log(main_loop)
        logger.info(" Model saved, took {} seconds.".format(time.time()-start))

        # Write the time and model path to the main loop
        # write the iteration count and the path to the model params.npz (note hardcoding of params.npz)
        main_loop.log.status['last_saved_model'] = (self.main_loop.log.status['iterations_done'], self.path_to_parameters)


    def do(self, callback_name, *args):
        try:
            self.dump(self.main_loop)
        except Exception:
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (self.path_to_folder +
                                                            'params.npz',))


class RunExternalValidation(SimpleExtension):
    """Look for the path to a checkpointed model (*.params) on the main_loop.log.status
       - if you find one, copy the config object and write the path of that model into (load_from_saved_params, saved_parameters)
       - note there is currently a disconnect between training mode and evaluation
       - spin up a new NMT model in evaluation mode, and run it with the saved params

        Saves only parameters (npz), iteration state (pickle) and log (pickle).

    """

    # TODO: hardcode or specify the command to run in a different thread (use subprocess syntax)
    # TODO: another extension which can write to main loop when new bests are found -- this is useful for live plotting
    # TODO: mkdir and copy params when models are good
    # TODO: each model has a unique name while it's being validated
    # TODO: only start validation at a reasonable time
    def __init__(self, config, **kwargs):

        # Create saving directory if it does not exist
        saveto = config['saveto']
        if not os.path.exists(saveto):
            os.makedirs(saveto)
        self.temp_evaluation_dir = os.path.join(saveto, 'temp_evaluation_dir')
        if not os.path.exists(self.temp_evaluation_dir):
            os.makedirs(self.temp_evaluation_dir)

        self.config = config
        self.which_gpu = str(config.get('dev_gpu', 1))

        # this is used to parse the evaluation result files
        self.eval_result_delimiter = ' '
        # this holds the names of the results we've already processed
        self.processed_results = set()

        # this maintains the state for the most recently validated model
        # this field is a tuple: (<iteration_count>, model_path)
        self.most_recent_validation = None

        self.metrics = config.get('evaluation_metrics', ['bleu'])
        self.results = {}
        for m in self.metrics:
            self.results[m] = []

        self.savedir = os.path.join(saveto, 'validation_models')
        # TODO: manually set which GPU this is going to run on

        super(RunExternalValidation, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        try:
            last_checkpoint = self.main_loop.log.status.get('last_saved_model', None)
            if last_checkpoint is not None:
                if self.most_recent_validation is None:
                    self.run_validation(last_checkpoint)
                elif last_checkpoint[0] > self.most_recent_validation[0]:
                    self.run_validation(last_checkpoint)
                else:
                    logger.info('Not running validation because there is no new model to evaluate')

            # check if any validations finished since the last time we ran this validator
            # since validations are asychronous, a later validation may finish before an earlier one
            self.check_for_validation_results()

        except Exception:
            raise
        finally:
            pass


    def build_evaluation_command(self, config_filename):
        """Note that machine_translation must be available as a module for this command to work"""
        command = ['python', '-m', 'machine_translation', '-m', 'evaluate', config_filename]
        return command


    def run_validation(self, last_checkpoint):
        """Run the validation in another thread on another GPU"""
        logger.info('RUNNING VALIDATION')

        # TODO: how to set environment CUDA_VISIBLE_DEVICES for the thread?

        temp_model_prefix = 'evaluation_checkpoint_{}'.format(last_checkpoint[0])

        temp_model_path = os.path.join(self.temp_evaluation_dir, temp_model_prefix + '.npz')

        # copy model to a temporary directory
        shutil.copy(last_checkpoint[1], temp_model_path)
        logger.info('copied model to: {}'.format(temp_model_path))

        # write config for model evaluation
        eval_config = copy.copy(self.config)
        eval_config['saved_parameters'] = temp_model_path

        # make the dev set look like the test set in the dynamic config so that we can run in evaluation mode
        eval_config['test_set'] = eval_config['val_set']
        eval_config['test_gold_refs'] = eval_config['val_set_grndtruth']
        eval_config['translated_output_file'] = os.path.join(self.temp_evaluation_dir, temp_model_prefix + '.out')
        eval_config['saveto'] = self.temp_evaluation_dir
        eval_config['model_name'] = temp_model_prefix

        eval_config_filename = os.path.join(self.temp_evaluation_dir, temp_model_prefix + '.config.yaml')
        # write config to file
        with codecs.open(eval_config_filename, 'w', encoding='utf8') as f:
            yaml.dump(eval_config, f, default_flow_style=False)

        evaluation_command = self.build_evaluation_command(eval_config_filename)

        new_env = os.environ.copy()
        new_env["CUDA_VISIBLE_DEVICES"] = self.which_gpu
        p = subprocess.Popen(evaluation_command, env=new_env)
        logger.info('Started validation in subprocess with command: {}'.format(' '.join(evaluation_command)))

        # indicate that this validation has been run for this iteration
        self.most_recent_validation = last_checkpoint


    def check_for_validation_results(self):
        evaluation_dir = os.path.join(self.temp_evaluation_dir, 'evaluation_reports')

        # look in the validation dir, read all results by name -- try to parse into our object which stores results per metric
        # TODO: also delete old models whose perf wasn't good so that disk usage doesn't go crazy
        # TODO: if there is a new validation which outperforms others, we should move it/rename it, etc...
        # if it is better, save it in a validation dir according to its metric
        if os.path.isdir(evaluation_dir):
            eval_files = [f for f in os.listdir(evaluation_dir) if os.path.isfile(os.path.join(evaluation_dir, f))]

            new_evaluations = [f for f in eval_files if f not in self.processed_results]
            for f in new_evaluations:
                self.processed_results.update([f])
                metric, score, name = f.split(self.eval_result_delimiter)
                self.results[metric].append((float(score), name))

            # the metrics sorted by iter count will give us the most recent result, add this to main loop for live plot
            for m in self.metrics:
                metric_results = self.results[m]
                if len(metric_results) > 0:
                    # sort by name -- the iter number is included in the name
                    metric_results = [r for r in sorted(metric_results, key=lambda x: x[1], reverse=True)]
                    # log the score of the most recent available validation result for this metric
                    self.main_loop.log.current_row['validation_set_{}_score'.format(m)] = metric_results[0][0]
                    # self.main_loop.log.current_row['validation_set_{}_score'.format(m)] = 20.0


    def copy_model(self, model_path, copy_path, new_name=None):
        """copy a model to a new path, optionally give it a new name"""
        pass


class LoadNMT(TrainingExtension, SaveLoadUtils):
    """Loads parameters log and iterations state."""

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        super(LoadNMT, self).__init__(saveto, **kwargs)

    def before_training(self):
        if not os.path.exists(self.path_to_folder):
            logger.info("No dump found")
            return
        logger.info("Loading the state from {} into the main loop"
                    .format(self.path_to_folder))
        try:
            self.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.path_to_folder
        except Exception:
            reraise_as("Failed to load the state")

    def load_parameters(self):
        return self.load_parameter_values(self.path_to_parameters)

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return load(source)

    def load_log(self):
        with open(self.path_to_log, "rb") as source:
            return cPickle.load(source)

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Reloading model")
        try:
            logger.info(" ...loading model parameters")
            params_all = self.load_parameters()
            self.set_model_parameters(main_loop.model, params_all)
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading iteration state...")
            main_loop.iteration_state = self.load_iteration_state()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading log...")
            main_loop.log = self.load_log()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))