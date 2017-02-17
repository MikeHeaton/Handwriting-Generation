from config import PARAMS
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import raw_rnn

def make_l1_and_window_layers(cell,
                                input_placeholder,
                                sequence_lengths_placeholder,
                                charlist_placeholder,
                                charlist_lengths_placeholder,
                                layer1_initial_state_placeholder,
                                kappa_initial_placeholder,
                                time_major=False):
    """
    # cell is the LSTM cell object. Required to create the zero-state at t=0.

    # input_placeholder is the location data for the sample.
    #   shape = [PARAMS.batch_size, PARAMS.sequence_len, 3]

    # sequence_lengths_placeholder contains the actual seqence lengths for each
    # sample in the batch.
    #   shape = [PARAMS.batch_size]

    # charlist_placeholder is the character lists for each sample, as integers.
    #   shape = [PARAMS.batch_size, PARAMS.max_char_len]

    # charlist_lengths_placeholder is the lengths for each character list.
    #   shape = [PARAMS.batch_size]

    # layer1_state_placeholder is the initial states placeholder for the LSTM cell.
    #    shape = [PARAMS.batch_size, PARAMS.lstm_size]

    # kappa_initial_placeholder is the initial values of kappa used to
    # initialise the window location.
        shape = [PARAMS.batch_size, PARAMS.window_gaussians]
    """

    def transform_h1_to_p(cell_output):
        W_h1_p = tf.get_variable("W_h1_p",
                        shape=(PARAMS.lstm_size, PARAMS.window_gaussians * 3),
                        initializer= tf.contrib.layers.xavier_initializer()
                        )
        b_p = tf.get_variable("b_p",
                        initializer= tf.zeros_initializer(shape=[PARAMS.window_gaussians * 3])
                        )

        # Get window parameters from the first output layer via matrix
        # multiplication. This is only for one time step, so
        # unlike in the model file there's no need to reshape cell_output.
        window_parameters = tf.matmul(cell_output, W_h1_p)
        window_parameters = tf.add(window_parameters, b_p)

        return window_parameters

    def find_parameters(window_parameters, current_loop_state):
        # Split the layer-p tensor into alpha, beta and kappa
        # and normalise them.
        window_parameters = tf.reshape(window_parameters,
                                       [batch_size,
                                        PARAMS.window_gaussians,
                                        3])
        (alphahat, betahat, kappahat) = (tf.squeeze(x, axis=2) for x in
                                         tf.split(2, 3, window_parameters))

        # Transform the *hat (pronounced *-hat) parameters into
        # proper distribution parameters, by normalising them as described in
        # https://arxiv.org/pdf/1308.0850v5.pdf page 26.
        alpha = tf.exp(alphahat)
        beta = tf.exp(betahat)

        print("Adding to make kappa:")
        print(current_loop_state, tf.exp(kappahat))
        kappa = tf.add(current_loop_state, tf.exp(kappahat))
        # Each param is now of shape [batch_size, PARAMS.window_gaussians].

        return alpha, beta, kappa

    def phi(u, alpha, beta, kappa):
        # u is a character position integer, ie we're looking at
        # the u'th character here and calculating its density.
        offsets = tf.sub(kappa, u)
        exponent = tf.mul(-beta, tf.square(offsets))
        exponential_term = tf.exp(exponent)
        density_per_gaussian = tf.mul(alpha, exponential_term)
        return tf.reduce_sum(density_per_gaussian, axis=1)

    def make_opinionvector(alpha, beta, kappa,
                            charlist_placeholder,
                            charlist_lengths_placeholder):
        # Calculate density for each character in sequence
        opinions_per_character = tf.stack([phi(u, alpha, beta, kappa)
                                           for u in
                                           range(PARAMS.max_char_len)],
                                          axis=1)

        # Mask out opinions about characters beyond the end of the sequences
        charlength_mask = tf.sequence_mask(charlist_lengths_placeholder,
                                            maxlen=PARAMS.max_char_len,
                                            dtype=tf.float32,
                                            name=None)
        opinions_per_character = tf.mul(opinions_per_character,
                                        charlength_mask)

        # charlist_placeholder is a 1-hot tensor for each character.
        # multiply it by opinions_per_character so that it's a p-hot tensor
        # instead, by tiling opinions_per_character downwards and Xing
        # elementwise.
        opinions_per_character = tf.expand_dims(opinions_per_character, 2)
        opinions_per_character = tf.tile(opinions_per_character, [1, 1, PARAMS.num_characters])
        opinionvectors = tf.mul(charlist_placeholder, opinions_per_character)

        # Reduce over the character sequence axis to get a total opinion
        # on which letter is being written at the present time.
        opinionvectors = tf.reduce_sum(opinionvectors, axis=1)

        return opinionvectors

    # Unpack assumes time is the 0'th dimension,
    # so permute input sequence to make it so.
    if not time_major:
        input_placeholder = tf.transpose(input_placeholder,
                                         perm=[1, 0, 2])

    # Get batch_size
    batch_size = input_placeholder.get_shape()[1].value
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)#PARAMS.sequence_len)
    inputs_ta = inputs_ta.unpack(input_placeholder)

    def window_loop_fn(time, cell_output, cell_state, loop_state):

        if cell_output is None:
            samples_finished = (time >= sequence_lengths_placeholder)

            next_offset_input = inputs_ta.read(time)
            opinion_vectors = tf.zeros([batch_size, PARAMS.num_characters])
            next_input = tf.zeros([batch_size, 3 + PARAMS.num_characters])

            #next_input = tf.concat(concat_dim=1,
            #                      values=[next_offset_input, opinion_vectors])

            next_cell_state = layer1_initial_state_placeholder
            emit_output = tf.zeros([PARAMS.num_characters+PARAMS.lstm_size])
            loop_state = kappa_initial_placeholder

            W_h1_p = tf.get_variable("W_h1_p",
                            shape=(PARAMS.lstm_size, PARAMS.window_gaussians * 3),
                            initializer= tf.contrib.layers.xavier_initializer()
                            )
            b_p = tf.get_variable("b_p",
                            initializer= tf.zeros_initializer(shape=[PARAMS.window_gaussians * 3])
                            )

            """TODO: better initialization for the opinionvector would be
            the first character of each sample. Does that make a difference?"""

            return samples_finished, next_input, next_cell_state, emit_output, loop_state
        else:
            next_cell_state = cell_state
            tf.get_variable_scope().reuse_variables()

            #with tf.name_scope("WINDOW_LAYER"):

            W_h1_p = tf.get_variable("W_h1_p",
                            shape=(PARAMS.lstm_size, PARAMS.window_gaussians * 3),
                            initializer= tf.contrib.layers.xavier_initializer()
                            )
            b_p = tf.get_variable("b_p",
                            initializer= tf.zeros_initializer(shape=[PARAMS.window_gaussians * 3])
                            )

            # Get window parameters from the first output layer via matrix
            # multiplication. This is only for one time step, so
            # unlike in the model file there's no need to reshape cell_output.
            window_parameters = tf.matmul(cell_output, W_h1_p)
            p_layer_inputs = tf.add(window_parameters, b_p)

            #p_layer_inputs = transform_h1_to_p(cell_output)
            alpha, beta, kappa = find_parameters(p_layer_inputs, loop_state)

            opinion_vectors = make_opinionvector(alpha, beta, kappa,
                                            charlist_placeholder,
                                            charlist_lengths_placeholder)

            # lstm_2 is fed the output from layer 1 and the opinion vector.
            """TODO: add skip connection by joining the current output."""
            emit_output = tf.concat(concat_dim=1,
                                  values=[cell_output,
                                          opinion_vectors])
            #emit_output = tf.squeeze(emit_output)

            #with tf.name_scope("LOOP_FN"):

            samples_finished = (time >= sequence_lengths_placeholder)
            loop_state = kappa

            # Concat the actual input with the opinionvector to get the input
            # to the RNN for time t+1.
            next_offset_input = inputs_ta.read(time-1)
            next_input = tf.concat(concat_dim=1,
                                  values=[next_offset_input, opinion_vectors])

            print("samples_finished", samples_finished)
            print("next_input", next_input)
            print("next_cell_state", next_cell_state)
            print("emit_output", emit_output)
            print("loop_state", loop_state)

            return samples_finished, next_input, next_cell_state, emit_output, loop_state

    outputs_ta, final_state, final_loop_state = raw_rnn(cell, window_loop_fn)
    print("raw_rnn called success")
    outputs = outputs_ta.pack()

    if not time_major:
        outputs = tf.transpose(outputs,
                                perm=[1, 0, 2])

    return outputs, final_state, final_loop_state


if __name__ == "__main__":
    """Run test with dummy tensors"""
    sequence_len = PARAMS.sequence_len
    input_placeholder = tf.placeholder(tf.float32,
                                shape=( PARAMS.batch_size,
                                        sequence_len,
                                        3))
    sequence_lengths_placeholder = tf.ones([PARAMS.batch_size], dtype=tf.int32)
    charlist_placeholder = tf.placeholder(tf.float32,
                                shape=( PARAMS.batch_size,
                                        PARAMS.max_char_len,
                                        PARAMS.num_characters))
    charlist_lengths_placeholder = tf.placeholder(tf.int32,
                                shape=( PARAMS.batch_size,))

    layer1_state_placeholder =  tf.placeholder(tf.float32,
                                shape=( PARAMS.batch_size, PARAMS.lstm_size * 2))
    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                    PARAMS.lstm_size,
                    state_is_tuple=False,
                    cell_clip=1,
                    initializer=tf.contrib.layers.xavier_initializer()
                    )

    kappa_initial_placeholder = tf.zeros([PARAMS.batch_size, PARAMS.window_gaussians])

    layeroutput = make_l1_and_window_layers(lstm_cell,
                                        input_placeholder,
                                        sequence_lengths_placeholder,
                                        charlist_placeholder,
                                        charlist_lengths_placeholder,
                                        layer1_state_placeholder,
                                        kappa_initial_placeholder,
                                        time_major=False)
