
from theano import tensor

# from blocks.bricks.sequence_generators import BaseSequenceGenerator
from machine_translation.model import Decoder
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, COST
from blocks.utils import dict_union, dict_subset

# A decoder which replaces the sequence generator cross entropy cost function with a minimum-risk training objective
# based on expectation over a sentence-level metric


# WORKING:
# subclass sequence generator and replace cost function
# remember that the sequence generator cost function delegates to readout, then to emitter
# For generation, we still want to use categorical cross-entropy from the softmax emitter
# readout.cost is called in sequence_generator.generate
"""Returns generation costs for output sequences.
@application
def cost_matrix(self, application_call, outputs, mask=None, **kwargs):

    See Also
    --------
    :meth:`cost` : Scalar cost.

"""


class MinimumRiskSequenceGenerator(Decoder):

    # def __init__(self, *args, **kwargs):
    #     super(MinimumRiskSequenceGenerator, self).__init__(*args, **kwargs)


    # TODO: can we override a method _and_ change its signature -- yes, but it breaks the Liskov substitution principle
    # WORKING: one solution to this is to put the scores into kwargs, but the semantics of 'samples' vs. 'outputs' is
    # still different, so maybe it's ok to change the signature
    # Note: the @application decorator inspects the arguments, and transparently adds args for 'application' and
    # 'application_call'
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_samples_mask', 'target_samples'],
                 outputs=['cost'])
    def cost(self, application_call, representation, source_sentence_mask,
             target_samples, target_samples_mask, **kwargs):
        # emulate the process in sequence_generator.cost_matrix, but compute log probabilities instead of costs
        # for each sample, we need its probability according to the model (these could actually be passed from the
        # sampling model, which could be more efficient)

        # Transpose everything (note we can use transpose here only if it's 2d)
        source_sentence_mask = source_sentence_mask.T

        # starting dimension of samples should be [batch, sample, time]
        # need to flatten out then reshape to keep interfaces consistent
        samples_shape = target_samples.shape
        samples = target_samples.reshape(samples_shape[0]*samples_shape[1], samples_shape[2])
        samples_mask = target_samples_mask.reshape((samples.shape))

        # make samples (time, batch)
        samples = samples.T
        samples_mask = samples_mask.T

        batch_size = samples.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        feedback = self.readout.feedback(samples)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=samples_mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(batch_size)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))

        # TODO: get the token probabilities instead of the costs _HERE_
        # TODO: multiply (logspace add) token probabilities to get sequence probabilities
        # TODO: compute the g() function for each sequence (vector of probs) with temperature

        # WORKING: return 0. cost and get the whole pipeline running

        # costs = self.readout.cost(readouts, outputs)
        # if mask is not None:
        #     costs *= mask
        #
        # for name, variable in list(glimpses.items()) + list(states.items()):
        #     application_call.add_auxiliary_variable(
        #         variable.copy(), name=name)

        # This variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        # for name in self._state_names + self._glimpse_names:
        #     application_call.add_auxiliary_variable(
        #         results[name][-1].copy(), name=name+"_final_value")

        # return costs

        # target_sentence = target_sentence.T
        # target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        # cost = self.sequence_generator.cost_matrix(**{
        #     'mask': target_sentence_mask,
        #     'outputs': target_sentence,
        #     'attended': representation,
        #     'attended_mask': source_sentence_mask}
        #                                            )

        # return (cost * target_sentence_mask).sum() / \
        #        target_sentence_mask.shape[1]
        return 0.

    # TODO: for generation, we don't want to use the sequence-level expected score, because we are generating
    # TODO: step-by-step, this would require some form of expected future cost to make sense

    @application
    def generate(self, source_sentence, representation, **kwargs):
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            **kwargs)
