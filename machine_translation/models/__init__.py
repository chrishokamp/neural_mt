from theano import tensor

from blocks.bricks import Initializable
from blocks.bricks.base import application, Brick, lazy
from blocks.bricks import NDimensionalSoftmax

from blocks.roles import add_role, COST
from blocks.utils import dict_union, dict_subset
from blocks.bricks.recurrent import recurrent
# from blocks.bricks.sequence_generators import BaseSequenceGenerator
from blocks.bricks.sequence_generators import SequenceGenerator

# A decoder which replaces the sequence generator cross entropy cost function with a minimum-risk training objective
# based on expectation over a sentence-level metric


# WORKING:
# subclass sequence generator and replace cost function
# remember that the sequence generator cost function delegates to readout, then to emitter
# For generation, we still want to use categorical cross-entropy from the softmax emitter
# readout.cost is called in sequence_generator.generate

# Working -- subclass sequence generator, and change the MT decoder so that we don't get into problems with the blocks
# application/call Hierarchy
# TODO: where/how does blocks determine the bricks' hierarchical names?
class MinRiskSequenceGenerator(SequenceGenerator):
    r"""A generic sequence generator.

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component of the sequence generator.
    transition : instance of :class:`AbstractAttentionRecurrent`
        The transition component of the sequence generator.
    fork : :class:`.Brick`
        The brick to compute the transition's inputs from the feedback.

    See Also
    --------
    :class:`.Initializable` : for initialization parameters

    :class:`SequenceGenerator` : more user friendly interface to this\
        brick

    """

    def __init__(self, *args, **kwargs):
        self.softmax = NDimensionalSoftmax()
        super(MinRiskSequenceGenerator, self).__init__(*args, **kwargs)
        self.children.append(self.softmax)

    @application
    def probs(self, readouts):
        return self.softmax.apply(readouts, extra_ndim=readouts.ndim - 2)


    # TODO: can we override a method _and_ change its signature -- yes, but it breaks the Liskov substitution principle
    # TODO: the shape of samples is stored in `scores`, look there to do the neccessary reshapes
    # WORKING: one solution to this is to put the scores into kwargs, but the semantics of 'samples' vs. 'outputs' is
    # still different, so maybe it's ok to change the signature
    # Note: the @application decorator inspects the arguments, and transparently adds args for 'application' and
    # 'application_call'
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_samples_mask', 'target_samples', 'scores'],
                 outputs=['cost'])
    def expected_cost(self, application_call, representation, source_sentence_mask,
             target_samples, target_samples_mask, scores, **kwargs):
        # emulate the process in sequence_generator.cost_matrix, but compute log probabilities instead of costs
        # for each sample, we need its probability according to the model (these could actually be passed from the
        # sampling model, which could be more efficient)

        # Transpose everything (note we can use transpose here only if it's 2d)
        source_sentence_mask = source_sentence_mask.T

        # starting dimension of samples should be [batch, sample, time]
        # need to flatten out then reshape to keep interfaces consistent
        # samples_shape = target_samples.shape
        # samples = target_samples.reshape((samples_shape[0]*samples_shape[1], samples_shape[2]))
        # samples_mask = target_samples_mask.reshape(samples.shape)

        # make samples (time, batch)
        samples = target_samples.T
        samples_mask = target_samples_mask.T

        # target_sentence = target_sentence.T
        # target_sentence_mask = target_sentence_mask.T

        # we need this to set the 'attended' kwarg
        keywords = {
            'mask': target_samples_mask,
            'outputs': target_samples,
            'attended': representation,
            'attended_mask': source_sentence_mask
        }

        # return (cost * target_sentence_mask).sum() / \
        #     target_sentence_mask.shape[1]


        batch_size = samples.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(keywords, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(keywords, self._context_names, must_have=False)
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

        # get the token probabilities -- should be shape = (time, batch, vocab)
        # the max at each timestep is the word that was chosen(?)
        #  -- No, we need the index of the word to know which was chosen
        # TODO: another option is to compute the sequence probability in the sampling step
        word_probs = self.probs(readouts)
        word_probs = tensor.log(word_probs)
        # * samples_mask


        # Note: converting the samples to one hot wastes space, but it gets the job done
        one_hot_samples = tensor.eye(word_probs.shape[-1])[samples]
        one_hot_samples.astype('float32')
        actual_probs = word_probs * one_hot_samples

        # reshape to (batch, time, prob), take elementwise log, then sum over the batch dimension
        # to get sequence-level probability
        actual_probs = actual_probs.dimshuffle(1,0,2)
        # we are first summing over vocabulary (only one non zero cell)
        sequence_probs = actual_probs.sum(axis=2)
        sequence_probs = sequence_probs * target_samples_mask
        # now sum over time dimension
        sequence_probs = sequence_probs.sum(axis=1)

        sequence_probs = sequence_probs.reshape(scores.shape)

        # now get the g function for each instance, by doing a softmax with temperature param
        # over each sample from that instance

        # TODO: reshape using scores.shape to get the g() distribution for each sequence


        # TODO: compute the g() function for each sequence (vector of probs) with temperature




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

	#return readouts.sum() * 0.
	#return tensor.zeros_like(representation).astype('float32').sum()

	# return representation.sum()
     #    return readouts.sum()
     #    return word_probs.sum()
     #    return sequence_probs.sum()
     #    return log_word_probs
     #    return word_probs
     #    return one_hot_samples
     #    return actual_probs
        return sequence_probs
     #    return target_samples
     #    return source_sentence_mask
        # return readouts


