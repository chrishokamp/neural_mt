
from theano import tensor

# from blocks.bricks.sequence_generators import BaseSequenceGenerator
from machine_translation.model import Decoder
from blocks.bricks import Initializable
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, COST
from blocks.utils import dict_union, dict_subset
from blocks.bricks.recurrent import recurrent

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


# TODO: probably need to make a MinimumRiskSequence generator instead, because we have this issue with
# TODO: calling through the brick hierarchy
class MinimumRiskSequenceDecoder(Decoder):

    def __init__(self, *args, **kwargs):
        super(MinimumRiskSequenceDecoder, self).__init__(*args, **kwargs)

        # HACK -- what happens with initialization when we hijack these from self.sequence_generator?
        # to
        # self.children.extend([self.sequence_generator.readout, self.sequence_generator.fork,
        #                      self.sequence_generator.transition])


    # def _push_initialization_config(self):
    #     for child in super(MinimumRiskSequenceDecoder, self).children:
    #         if isinstance(child, Initializable):
    #             child.rng = self.rng
    #             if self.weights_init:
    #                 child.weights_init = self.weights_init
    #     if hasattr(self, 'biases_init') and self.biases_init:
    #         for child in self.children:
    #             if (isinstance(child, Initializable) and
    #                     hasattr(child, 'biases_init')):
    #                 child.biases_init = self.biases_init

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
        # samples_shape = target_samples.shape
        # samples = target_samples.reshape((samples_shape[0]*samples_shape[1], samples_shape[2]))
        # samples_mask = target_samples_mask.reshape(samples.shape)
        samples = target_samples
        samples_mask = target_samples_mask

        source_sentence_mask = source_sentence_mask.T
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

        # make samples (time, batch)
        samples = samples.T
        samples_mask = samples_mask.T

        batch_size = samples.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(keywords, self.sequence_generator._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(keywords, self.sequence_generator._context_names, must_have=False)
        feedback = self.sequence_generator.readout.feedback(samples)
        inputs = self.sequence_generator.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.sequence_generator.transition.apply(
            mask=samples_mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self.sequence_generator._state_names}
        glimpses = {name: results[name][1:] for name in self.sequence_generator._glimpse_names}

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.sequence_generator.readout.feedback(self.sequence_generator.readout.initial_outputs(batch_size)))
        readouts = self.sequence_generator.readout.readout(
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
        return readouts.sum() * 0.

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



# Working -- subclass sequence generator, and change the MT decoder so that we don't get into problems with the blocks
# Hierarchy
class BaseSequenceGenerator(Initializable):
    r"""A generic sequence generator.

    This class combines two components, a readout network and an
    attention-equipped recurrent transition, into a context-dependent
    sequence generator. Third component must be also given which
    forks feedback from the readout network to obtain inputs for the
    transition.

    The class provides two methods: :meth:`generate` and :meth:`cost`. The
    former is to actually generate sequences and the latter is to compute
    the cost of generating given sequences.

    The generation algorithm description follows.

    **Definitions and notation:**

    * States :math:`s_i` of the generator are the states of the transition
      as specified in `transition.state_names`.

    * Contexts of the generator are the contexts of the
      transition as specified in `transition.context_names`.

    * Glimpses :math:`g_i` are intermediate entities computed at every
      generation step from states, contexts and the previous step glimpses.
      They are computed in the transition's `apply` method when not given
      or by explicitly calling the transition's `take_glimpses` method. The
      set of glimpses considered is specified in
      `transition.glimpse_names`.

    * Outputs :math:`y_i` are produced at every step and form the output
      sequence. A generation cost :math:`c_i` is assigned to each output.

    **Algorithm:**

    1. Initialization.

       .. math::

           y_0 = readout.initial\_outputs(contexts)\\
           s_0, g_0 = transition.initial\_states(contexts)\\
           i = 1\\

       By default all recurrent bricks from :mod:`~blocks.bricks.recurrent`
       have trainable initial states initialized with zeros. Subclass them
       or :class:`~blocks.bricks.recurrent.BaseRecurrent` directly to get
       custom initial states.

    2. New glimpses are computed:

       .. math:: g_i = transition.take\_glimpses(
           s_{i-1}, g_{i-1}, contexts)

    3. A new output is generated by the readout and its cost is
       computed:

       .. math::

            f_{i-1} = readout.feedback(y_{i-1}) \\
            r_i = readout.readout(f_{i-1}, s_{i-1}, g_i, contexts) \\
            y_i = readout.emit(r_i) \\
            c_i = readout.cost(r_i, y_i)

       Note that the *new* glimpses and the *old* states are used at this
       step. The reason for not merging all readout methods into one is
       to make an efficient implementation of :meth:`cost` possible.

    4. New states are computed and iteration is done:

       .. math::

           f_i = readout.feedback(y_i) \\
           s_i = transition.compute\_states(s_{i-1}, g_i,
                fork.apply(f_i), contexts) \\
           i = i + 1

    5. Back to step 2 if the desired sequence
       length has not been yet reached.

    | A scheme of the algorithm described above follows.

    .. image:: /_static/sequence_generator_scheme.png
            :height: 500px
            :width: 500px

    ..

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
    @lazy()
    def __init__(self, readout, transition, fork, **kwargs):
        super(BaseSequenceGenerator, self).__init__(**kwargs)
        self.readout = readout
        self.transition = transition
        self.fork = fork

        self.children = [self.readout, self.fork, self.transition]

    @property
    def _state_names(self):
        return self.transition.compute_states.outputs

    @property
    def _context_names(self):
        return self.transition.apply.contexts

    @property
    def _glimpse_names(self):
        return self.transition.take_glimpses.outputs

    def _push_allocation_config(self):
        # Configure readout. That involves `get_dim` requests
        # to the transition. To make sure that it answers
        # correctly we should finish its configuration first.
        self.transition.push_allocation_config()
        transition_sources = (self._state_names + self._context_names +
                              self._glimpse_names)
        self.readout.source_dims = [self.transition.get_dim(name)
                                    if name in transition_sources
                                    else self.readout.get_dim(name)
                                    for name in self.readout.source_names]

        # Configure fork. For similar reasons as outlined above,
        # first push `readout` configuration.
        self.readout.push_allocation_config()
        feedback_name, = self.readout.feedback.outputs
        self.fork.input_dim = self.readout.get_dim(feedback_name)
        self.fork.output_dims = self.transition.get_dims(
            self.fork.apply.outputs)

    # TODO: can we override a method _and_ change its signature -- yes, but it breaks the Liskov substitution principle
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
        samples = target_samples
        samples_mask = target_samples_mask

        source_sentence_mask = source_sentence_mask.T
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

        # make samples (time, batch)
        samples = samples.T
        samples_mask = samples_mask.T

        batch_size = samples.shape[1]

        # Prepare input for the iterative part
        #states = dict_subset(keywords, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        #contexts = dict_subset(keywords, self._context_names, must_have=False)
        #feedback = self.readout.feedback(samples)
        #inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        #results = self.transition.apply(
        #    mask=samples_mask, return_initial_states=True, as_dict=True,
        #    **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        #states = {name: results[name][:-1] for name in self._state_names}
        #glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        #feedback = tensor.roll(feedback, 1, 0)
        #feedback = tensor.set_subtensor(
        #    feedback[0],
        #    self.readout.feedback(self.readout.initial_outputs(batch_size)))
        #readouts = self.readout.readout(
        #    feedback=feedback, **dict_union(states, glimpses, contexts))



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

	#return readouts.sum() * 0.
	#return tensor.zeros_like(representation).astype('float32').sum()

	return representation.sum()


    @application
    def cost(self, application_call, outputs, mask=None, **kwargs):
        """Returns the average cost over the minibatch.

        The cost is computed by averaging the sum of per token costs for
        each sequence over the minibatch.

        .. warning::
            Note that, the computed cost can be problematic when batches
            consist of vastly different sequence lengths.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The 3(2) dimensional tensor containing output sequences.
            The axis 0 must stand for time, the axis 1 for the
            position in the batch.
        mask : :class:`~tensor.TensorVariable`
            The binary matrix identifying fake outputs.

        Returns
        -------
        cost : :class:`~tensor.Variable`
            Theano variable for cost, computed by summing over timesteps
            and then averaging over the minibatch.

        Notes
        -----
        The contexts are expected as keyword arguments.

        Adds average cost per sequence element `AUXILIARY` variable to
        the computational graph with name ``per_sequence_element``.

        """
        # Compute the sum of costs
        costs = self.cost_matrix(outputs, mask=mask, **kwargs)
        cost = tensor.mean(costs.sum(axis=0))
        add_role(cost, COST)

        # Add auxiliary variable for per sequence element cost
        application_call.add_auxiliary_variable(
            (costs.sum() / mask.sum()) if mask is not None else costs.mean(),
            name='per_sequence_element')
        return cost

    @application
    def cost_matrix(self, application_call, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.

        See Also
        --------
        :meth:`cost` : Scalar cost.

        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
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
        costs = self.readout.cost(readouts, outputs)
        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)

        # This variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        for name in self._state_names:
            application_call.add_auxiliary_variable(
                results[name][-1].copy(), name=name+"_final_value")

        return costs

    @recurrent
    def generate(self, outputs, **kwargs):
        """A sequence generation step.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The outputs from the previous step.

        Notes
        -----
        The contexts, previous states and glimpses are expected as keyword
        arguments.

        """
        states = dict_subset(kwargs, self._state_names)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        glimpses = dict_subset(kwargs, self._glimpse_names)

        next_glimpses = self.transition.take_glimpses(
            as_dict=True, **dict_union(states, glimpses, contexts))
        next_readouts = self.readout.readout(
            feedback=self.readout.feedback(outputs),
            **dict_union(states, next_glimpses, contexts))
        next_outputs = self.readout.emit(next_readouts)
        next_costs = self.readout.cost(next_readouts, next_outputs)
        next_feedback = self.readout.feedback(next_outputs)
        next_inputs = (self.fork.apply(next_feedback, as_dict=True)
                       if self.fork else {'feedback': next_feedback})
        next_states = self.transition.compute_states(
            as_list=True,
            **dict_union(next_inputs, states, next_glimpses, contexts))
        return (next_states + [next_outputs] +
                list(next_glimpses.values()) + [next_costs])

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('states')
    def generate_states(self):
        return self._state_names + ['outputs'] + self._glimpse_names

    @generate.property('outputs')
    def generate_outputs(self):
        return (self._state_names + ['outputs'] +
                self._glimpse_names + ['costs'])

    def get_dim(self, name):
        if name in (self._state_names + self._context_names +
                    self._glimpse_names):
            return self.transition.get_dim(name)
        elif name == 'outputs':
            return self.readout.get_dim(name)
        return super(BaseSequenceGenerator, self).get_dim(name)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        # TODO: support dict of outputs for application methods
        # to simplify this code.
        state_dict = dict(
            self.transition.initial_states(
                batch_size, as_dict=True, *args, **kwargs),
            outputs=self.readout.initial_outputs(batch_size))
        return [state_dict[state_name]
                for state_name in self.generate.states]

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.generate.states
