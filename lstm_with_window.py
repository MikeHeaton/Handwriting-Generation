from config import PARAMS
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs


class LayerOneAndWindowCell(rnn_cell_impl.RNNCell):
    """L1+Window cell which handles the window calculation, and the pass
    of the window results down-layer to L1 at the next time step
    (which is the hard bit)
         /-\
      /|  |
      /  WIND    WIND
     /   /-\  \  /-\
    |     |    \  |
     \    |     \ |
      \- L1 ---> L1
        /-\    /-\
         |      |    etc.
    """

    def __init__(self, cell, charseqs, reuse=False):
        """Init with:
        - L1Cell; and
        - a tensor of character sequences,
            of size [batch_size, max_char_len, alphabet_size]."""
        super().__init__(cell, _reuse=reuse)
        self.l1cell = cell
        self.charseqs = charseqs

    def zero_state(self, batch_size, dtype):
        return [self.l1cell.zero_state(batch_size, dtype),
                tf.zeros([batch_size, PARAMS.window_gaussians], dtype=dtype)]

    @property
    def state_size(self):
        return (self.l1cell.state_size, [PARAMS.window_gaussians])

    @property
    def output_size(self):
        return [self.l1cell.output_size, PARAMS.num_characters]

    def _l1_to_greeks(self, x_l1_output, prev_kappa):
        """
        Input: - the output from l1 at a time step, shape [batch_size, 3K]; and
               - previous kappa values, ie the model's assumption about
                 current character position,
                 size [batch_size, K].

                 (K = PARAMS.window_gaussians)

        Calculates and returns the greeks alpha, beta, kappa
        which determine the window parameters, as defined in p26 of
        https://arxiv.org/pdf/1308.0850v5.pdf.
        """
        # Reshape the output and split it into alpha, beta, kappa -hats.
        # (-hat designates raw, unnormalised params)
        x_l1_output = tf.reshape(x_l1_output,
                                 [-1,
                                  PARAMS.window_gaussians,
                                  3])
        (alphahat, betahat, kappahat) = (tf.squeeze(x, axis=2) for x in
                                         tf.split(x_l1_output, 3, axis=2))

        # Transform the *-hat parameters into
        # proper distribution parameters, by normalising them
        alpha = tf.exp(alphahat)
        beta = tf.exp(betahat)
        kappa = tf.add(prev_kappa, tf.exp(kappahat))
        # Each param is now of shape [batch_size, PARAMS.window_gaussians].

        return alpha, beta, kappa

    def _abk_to_w(self, alpha, beta, kappa):
        """
        Input: (alpha, beta, kappa) parameters, sizes [batch_size, K]

        Calculates and returns w, a vector of size [batch_size, alphabet_size]
        which encodes the model's belief that it's
        writing each character of the alphabet.
        See https://arxiv.org/pdf/1308.0850v5.pdf p26.

        We calculate phi for each character u then combine them.
        The charseqs tensor is a fixed size, but variable length sequences
        are managed automatically because  charseqs is 0-hot
        when we're beyond the actual sequence finish point.
        """

        all_phi_values = [self._phi(u, alpha, beta, kappa)
                          for u in range(PARAMS.max_char_len)]
        all_char_vectors = tf.unstack(self.charseqs, axis=1)

        weighted_chars = [tf.transpose(tf.multiply(p, tf.transpose(c)))
                          for p, c in zip(all_phi_values, all_char_vectors)]

        return tf.add_n(weighted_chars)

    def _phi(self, u, alpha, beta, kappa):
        # u is a character position integer, ie we're looking at
        # the u'th character here and calculating its density.
        offsets = tf.subtract(kappa, u)
        exponent = tf.multiply(-beta, tf.square(offsets))
        exponential_term = tf.exp(exponent)
        density_per_gaussian = tf.multiply(alpha, exponential_term)
        return tf.reduce_sum(density_per_gaussian, axis=1)

    def call(self, layer_input, state):
        """Input into l1 cell becomes concatenation of stroke input
        and the state from the previous window cell."""
        with vs.variable_scope("l1"):
            l1_prev_state, prev_kappa = state
            l1_input = array_ops.concat(
                [layer_input, prev_kappa], axis=1, name='input_to_cell')

            l1_output, l1_state = self.l1cell(l1_input, l1_prev_state)

        """Input into window cell becomes the character data"""
        with vs.variable_scope("window"):
            alpha, beta, kappa = self._l1_to_greeks(l1_output, prev_kappa)
            window_output = self._abk_to_w(alpha, beta, kappa)
        return [l1_output, window_output], [l1_state, kappa]


class WindowedMultiRNNCell(rnn_cell_impl.RNNCell):
    def __init__(self, l1windowcell, othercells, reuse=False):
        """Init with:
        - a LayerOneAndWindowCell (see above); and
        - a list of other rnncells to stack"""
        super().__init__(l1windowcell, _reuse=reuse)
        self.l1windowcell = l1windowcell
        self.othercells = othercells

    def zero_state(self, batch_size, dtype):
        return (self.l1windowcell.zero_state(batch_size, dtype) +
                [cell.zero_state(batch_size, dtype)
                 for cell in self.othercells])

    @property
    def state_size(self):
        return (list(self.l1windowcell.state_size) +
                [cell.state_size for cell in self.othercells])

    @property
    def output_size(self):
        return [self.l1windowcell.output_size +
                [cell.output_size for cell in self.othercells]]

    def call(self, all_layers_input, all_state):
        stroke_input = all_layers_input
        l1window_prevstate = [all_state[0], all_state[1]]

        l1window_output, output_state = self.l1windowcell(stroke_input,
                                                          l1window_prevstate)

        prewindow_output, layer_output = l1window_output
        all_outputs = l1window_output
        self.all_layer_inputs = [stroke_input]

        for i, cell in enumerate(self.othercells):
            with vs.variable_scope("l{}".format(i+2)):
                layer_input_state = all_state[i+2]
                # Other cells take skip connections from the input.
                layer_input = array_ops.concat(
                    [stroke_input, layer_output], axis=1,
                    name='input_to_cell_{}'.format(i+2))
                self.all_layer_inputs.append(layer_input)

                layer_output, layer_output_state = cell(layer_input,
                                                        layer_input_state)

                output_state.append(layer_output_state)
                all_outputs.append(layer_output)

        return all_outputs, output_state

        """Input into l1 cell becomes concatenation of stroke input
        and the state from the previous window cell."""
        with vs.variable_scope("l1"):
            l1_prev_state, prev_kappa = state
            l1_input = array_ops.concat(
                [layer_input, prev_kappa], axis=1, name='input_to_cell')

            l1_output, l1_state = self.l1cell(l1_input, l1_prev_state)

        """Input into window cell becomes the character data"""
        with vs.variable_scope("window"):
            alpha, beta, kappa = self._l1_to_greeks(l1_output, prev_kappa)
            window_output = self._abk_to_w(alpha, beta, kappa)
        return [l1_output, window_output], [l1_state, kappa]


def l1windowlayers_test():
    with tf.Session() as sess:
        """SET UP TEST"""
        input_placeholder = tf.placeholder(tf.float32,
                                           shape=(PARAMS.batch_size,
                                                  PARAMS.sequence_len,
                                                  3))
        char_placeholder = tf.placeholder(tf.int32,
                                          shape=(PARAMS.batch_size,
                                                 PARAMS.max_char_len))
        characters_1hot = tf.one_hot(char_placeholder,
                                     depth=PARAMS.num_characters,
                                     on_value=1.0,
                                     off_value=0.0,
                                     dtype=tf.float32)

        l1cell = tf.nn.rnn_cell.LSTMCell(
                        3 * PARAMS.window_gaussians,
                        state_is_tuple=False,
                        cell_clip=1,
                        initializer=tf.contrib.layers.xavier_initializer()
                        )

        dualcell = LayerOneAndWindowCell(l1cell, characters_1hot, reuse=False)
        zerostate = dualcell.zero_state(PARAMS.batch_size, tf.float32)

        rnn_outputs, last_state = tf.nn.dynamic_rnn(
                        cell=dualcell,
                        inputs=input_placeholder,
                        initial_state=zerostate,
                        dtype=tf.float32
                        )

        print("l1window setup ok :-)")

        """RUN TEST"""
        dummystrokes = np.ones([PARAMS.batch_size, PARAMS.sequence_len, 3])
        dummychars = np.ones([PARAMS.batch_size, PARAMS.max_char_len])

        sess.run(tf.global_variables_initializer())
        output, state = sess.run([rnn_outputs, last_state],
                          feed_dict={input_placeholder: dummystrokes,
                                     char_placeholder: dummychars})

        print("Output: ", [i.shape for i in output])
        print("State: ", [i.shape for i in state])
        print("l1window cell runs ok :-D")

def alllayers_test():
    with tf.Session() as sess:
        """SET UP TEST"""
        input_placeholder = tf.placeholder(tf.float32,
                                           shape=(PARAMS.batch_size,
                                                  PARAMS.sequence_len,
                                                  3))
        char_placeholder = tf.placeholder(tf.int32,
                                          shape=(PARAMS.batch_size,
                                                 PARAMS.max_char_len))
        characters_1hot = tf.one_hot(char_placeholder,
                                     depth=PARAMS.num_characters,
                                     on_value=1.0,
                                     off_value=0.0,
                                     dtype=tf.float32)

        l1cell = tf.nn.rnn_cell.LSTMCell(
                        3 * PARAMS.window_gaussians,
                        state_is_tuple=False,
                        cell_clip=1,
                        initializer=tf.contrib.layers.xavier_initializer()
                        )

        l2cell = tf.nn.rnn_cell.LSTMCell(
                        PARAMS.lstm_size,
                        state_is_tuple=False,
                        cell_clip=1,
                        initializer=tf.contrib.layers.xavier_initializer()
                        )

        l3cell = tf.nn.rnn_cell.LSTMCell(
                        PARAMS.lstm_size,
                        state_is_tuple=False,
                        cell_clip=1,
                        initializer=tf.contrib.layers.xavier_initializer()
                        )

        dualcell = LayerOneAndWindowCell(l1cell, characters_1hot, reuse=False)
        postwindowcells = [l2cell, l3cell]

        alllayerscell = WindowedMultiRNNCell(dualcell, postwindowcells)

        zerostate = alllayerscell.zero_state(PARAMS.batch_size, tf.float32)

        rnn_outputs, last_state = tf.nn.dynamic_rnn(
                        cell=alllayerscell,
                        inputs=input_placeholder,
                        initial_state=zerostate,
                        dtype=tf.float32
                        )

        print("allwindow setup ok ^_^")

        """RUN TEST"""
        dummystrokes = np.ones([PARAMS.batch_size, PARAMS.sequence_len, 3])
        dummychars = np.ones([PARAMS.batch_size, PARAMS.max_char_len])

        sess.run(tf.global_variables_initializer())
        output, state = sess.run([rnn_outputs, last_state],
                          feed_dict={input_placeholder: dummystrokes,
                                     char_placeholder: dummychars})
        print("allwindow cell runs ok \\^o^/")
        print("Inputs: ", [i.shape for i in alllayerscell.all_layer_inputs])
        print("Output: ", [j.shape for i in output for j in i])
        print("State: ", [i.shape for i in state])
        print("^[]^")


if __name__ == "__main__":
    """Run test with dummy tensors"""
    #l1windowlayers_test()

    alllayers_test()
