
from theano import tensor
from toolz import merge

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,
    SequenceGenerator)
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans

from machine_translation.models import MinRiskSequenceGenerator

from picklable_itertools.extras import equizip


# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


class LookupFeedbackWMT15(LookupFeedback):
    """Zero-out initial readout feedback by checking its value."""

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0

        shp = [outputs.shape[i] for i in range(outputs.ndim)]
        outputs_flat = outputs.flatten()
        outputs_flat_zeros = tensor.switch(outputs_flat < 0, 0,
                                           outputs_flat)

        lookup_flat = tensor.switch(
            outputs_flat[:, None] < 0,
            tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
            self.lookup.apply(outputs_flat_zeros))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup


class BidirectionalWMT15(Bidirectional):
    """Wrap two Gated Recurrents each having separate parameters."""

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]


class BidirectionalEncoder(Initializable):
    """Encoder of RNNsearch model."""

    def __init__(self, vocab_size, embedding_dim, state_dim, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim

        self.lookup = LookupTable(name='embeddings')
        self.bidir = BidirectionalWMT15(
            GatedRecurrent(activation=Tanh(), dim=state_dim))
        self.fwd_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='back_fork')

        self.children = [self.lookup, self.bidir,
                         self.fwd_fork, self.back_fork]

    def _push_allocation_config(self):
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim

        self.fwd_fork.input_dim = self.embedding_dim
        self.fwd_fork.output_dims = [self.bidir.children[0].get_dim(name)
                                     for name in self.fwd_fork.output_names]
        self.back_fork.input_dim = self.embedding_dim
        self.back_fork.output_dims = [self.bidir.children[1].get_dim(name)
                                      for name in self.back_fork.output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation'])
    def apply(self, source_sentence, source_sentence_mask):
        # Time as first dimension
        source_sentence = source_sentence.T
        source_sentence_mask = source_sentence_mask.T

        embeddings = self.lookup.apply(source_sentence)

        representation = self.bidir.apply(
            merge(self.fwd_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask}),
            merge(self.back_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask})
        )
        return representation


class GRUInitialState(GatedRecurrent):
    """Gated Recurrent with special initial state.

    Initial state of Gated Recurrent is set by an MLP that conditions on the
    last hidden state of the bidirectional encoder, applies an affine
    transformation followed by a tanh non-linearity to set initial state.

    """
    def __init__(self, attended_dim, **kwargs):
        super(GRUInitialState, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.initial_transformer = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        initial_state = self.initial_transformer.apply(
            attended[0, :, -self.attended_dim:])
        return initial_state

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)


# WORKING HERE -- initialize decoder initial state with constraint representation vector
class GRUSpecialInitialState(GatedRecurrent):
    """Gated Recurrent with user-specified initial state

    Initial state of Gated Recurrent is set by an MLP that conditions on the
    concatenated last hidden states of any bidirectional encoder, applies an affine
    transformation followed by a tanh non-linearity to set initial state.

    This differs from GRUInitialState because the user can provide any representation for the initial state

    """
    def __init__(self, representation_dim, representation_name='initial_state_representation',
                 **kwargs):
        super(GRUSpecialInitialState, self).__init__(**kwargs)
        self.representation_dim = representation_dim
        self.representation_name = representation_name
        self.initial_transformer = MLP(activations=[Tanh()],
                                       dims=[representation_dim, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        representation = kwargs[self.representation_name]
        # Note: this is the left-to-right representation only
        initial_state = self.initial_transformer.apply(
            representation[0, :, -self.representation_dim:])
        return initial_state

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                                                  name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)


class Decoder(Initializable):
    """
    Decoder of RNNsearch model.

    Parameters:
    -----------
    vocab_size: int
    embedding_dim: int
    representation_dim: int
    theano_seed: int
    loss_function: str : {'cross_entropy'(default) | 'min_risk'}

    """

    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, theano_seed=None, loss_function='cross_entropy', **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state
        self.transition = GRUInitialState(
            attended_dim=state_dim, dim=state_dim,
            activation=Tanh(), name='decoder')

        # Initialize the attention mechanism
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim, name="attention")

        # Initialize the readout, note that SoftmaxEmitter emits -1 for
        # initial outputs which is used by LookupFeedBackWMT15
        readout = Readout(
            source_names=['states', 'feedback',
                          # Chris: it's key that we're taking the first output of self.attention.take_glimpses.outputs
                          # Chris: the first output is the weighted avgs, the second is the weights in (batch, time)
                          self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=-1, theano_seed=theano_seed),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=state_dim, name='maxout_bias').apply,
                 Maxout(num_pieces=2, name='maxout').apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False, name='softmax0').apply,
                 Linear(input_dim=embedding_dim, name='softmax1').apply]),
            merged_dim=state_dim)

        # Build sequence generator accordingly
        if loss_function == 'cross_entropy':
            self.sequence_generator = SequenceGenerator(
                readout=readout,
                transition=self.transition,
                attention=self.attention,
                fork=Fork([name for name in self.transition.apply.sequences
                           if name != 'mask'], prototype=Linear())
            )
        elif loss_function == 'min_risk':
            self.sequence_generator = MinRiskSequenceGenerator(
                readout=readout,
                transition=self.transition,
                attention=self.attention,
                fork=Fork([name for name in self.transition.apply.sequences
                           if name != 'mask'], prototype=Linear())
            )
            # the name is important, because it lets us match the brick hierarchy names for the vanilla SequenceGenerator
            # to load pretrained models
            self.sequence_generator.name = 'sequencegenerator'
        else:
            raise ValueError('The decoder does not support the loss function: {}'.format(loss_function))

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': source_sentence_mask}
        )

        return (cost * target_sentence_mask).sum() / \
            target_sentence_mask.shape[1]
            # WORKING: try word-level cost
            # target_sentence_mask.sum()

    # Note: this requires the decoder to be using sequence_generator which implements expected cost
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_samples_mask', 'target_samples', 'scores'],
                 outputs=['cost'])
    def expected_cost(self, representation, source_sentence_mask, target_samples, target_samples_mask, scores,
                      **kwargs):
        return self.sequence_generator.expected_cost(representation,
                                                     source_sentence_mask,
                                                     target_samples, target_samples_mask, scores, **kwargs)


    @application
    def generate(self, source_sentence, representation, **kwargs):
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            **kwargs)
