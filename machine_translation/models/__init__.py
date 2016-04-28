from theano import tensor

from blocks.bricks.base import application
from blocks.bricks import NDimensionalSoftmax

from blocks.utils import dict_union, dict_subset
from blocks.bricks.sequence_generators import SequenceGenerator


class MinRiskSequenceGenerator(SequenceGenerator):
    r"""
    A sequence generator which replaces the cross-entropy cost function with a minimum-risk training objective
    based on expectation over a sentence-level metric

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

    """

    def __init__(self, *args, **kwargs):
        self.softmax = NDimensionalSoftmax()
        super(MinRiskSequenceGenerator, self).__init__(*args, **kwargs)
        self.children.append(self.softmax)

    @application
    def probs(self, readouts):
        return self.softmax.apply(readouts, extra_ndim=readouts.ndim - 2)


    # Note: the @application decorator inspects the arguments, and transparently adds args  ('application_call')
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_samples_mask', 'target_samples', 'scores'],
                 outputs=['cost'])
    def expected_cost(self, application_call, representation, source_sentence_mask,
             target_samples, target_samples_mask, scores, smoothing_constant=0.005, **kwargs):
        """
        emulate the process in sequence_generator.cost_matrix, but compute log probabilities instead of costs
        for each sample, we need its probability according to the model (these could actually be passed from the
        sampling model, which could be more efficient)
        """

        # Transpose everything (note we can use transpose here only if it's 2d, otherwise we need dimshuffle)
        source_sentence_mask = source_sentence_mask.T

        # make samples (time, batch)
        samples = target_samples.T
        samples_mask = target_samples_mask.T

        # we need this to set the 'attended' kwarg
        keywords = {
            'mask': target_samples_mask,
            'outputs': target_samples,
            'attended': representation,
            'attended_mask': source_sentence_mask
        }

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

        word_probs = self.probs(readouts)
        word_probs = tensor.log(word_probs)

        # Note: converting the samples to one-hot wastes space, but it gets the job done
        one_hot_samples = tensor.eye(word_probs.shape[-1])[samples]
        one_hot_samples.astype('float32')
        actual_probs = word_probs * one_hot_samples

        # reshape to (batch, time, prob), then sum over the batch dimension
        # to get sequence-level probability
        actual_probs = actual_probs.dimshuffle(1,0,2)
        # we are first summing over vocabulary (only one non-zero cell per row)
        sequence_probs = actual_probs.sum(axis=2)
        sequence_probs = sequence_probs * target_samples_mask
        # now sum over time dimension
        sequence_probs = sequence_probs.sum(axis=1)

        # reshape and do exp() to get the true probs back
        # sequence_probs = tensor.exp(sequence_probs.reshape(scores.shape))
        sequence_probs = sequence_probs.reshape(scores.shape)

        # Note that the smoothing constant can be set by user
        sequence_distributions = (tensor.exp(sequence_probs*smoothing_constant) /
                                  tensor.exp(sequence_probs*smoothing_constant)
                                  .sum(axis=1, keepdims=True))

        # the following lines are done explicitly for code clarity
        # -- first get sequence expectation, then sum up the expectations for every
        # seq in the minibatch
        expected_scores = (sequence_distributions * scores).sum(axis=1)
        expected_scores = expected_scores.sum(axis=0)

        return expected_scores

