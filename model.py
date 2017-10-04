
from config import PARAMS
import tensorflow as tf
import numpy as np
import lstm_with_window
from tensorflow.python.ops import array_ops


class HandwritingModel:
    def __init__(self, generate_mode=False):
        def dropoutwrap(cell):
            if (not self.generate_mode) and PARAMS.dropout_keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                                cell,
                                output_keep_prob=PARAMS.dropout_keep_prob)
            return cell

        self.generate_mode = generate_mode
        if self.generate_mode:
            self.batch_size = 1
            self.sequence_len = 1
        else:
            self.batch_size = PARAMS.batch_size
            self.sequence_len = PARAMS.sequence_len

        with tf.name_scope("PLACEHOLDERS"):
            self.input_placeholder = tf.placeholder(tf.float32,
                                                    shape=(self.batch_size,
                                                           None,
                                                           3))

            self.next_inputs = tf.placeholder(
                                                        tf.float32,
                                                        shape=(self.batch_size,
                                                               None,
                                                               3))

            self.inputs_length_placeholder = tf.placeholder(
                                                        tf.int32,
                                                        shape=(self.batch_size))

            """TODO: FIX SHAPE"""
            self.rnn_initial_state_placeholder = tf.placeholder(
                            tf.float32,
                            shape=(self.batch_size,
                                   PARAMS.lstm_size * 2 *
                                   PARAMS.number_of_postwindow_layers),
                            name="rnn_initial_state_placeholder")

            self.lr_placeholder = tf.placeholder(tf.float32,
                                                 name="lr_placeholder")

            self.character_codes_placeholder = tf.placeholder(
                                        tf.int32,
                                        shape=(self.batch_size,
                                               PARAMS.max_char_len),
                                        name="character_codes_placeholder")

            self.character_lengths_placeholder = tf.placeholder(
                                        tf.int32,
                                        shape=(self.batch_size),
                                        name="character_lengths_placeholder")

            characters_1hot = tf.one_hot(self.character_codes_placeholder,
                                         depth=PARAMS.num_characters,
                                         on_value=1.0,
                                         off_value=0.0,
                                         dtype=tf.float32)

        with tf.name_scope("LSTM_LAYERS"):
            l1cell = tf.nn.rnn_cell.LSTMCell(
                            3 * PARAMS.window_gaussians,
                            state_is_tuple=False,
                            # cell_clip=1,
                            initializer=tf.contrib.layers.xavier_initializer()
                            )
            l1cell = dropoutwrap(l1cell)

            dualcell = lstm_with_window.LayerOneAndWindowCell(l1cell,
                                                              characters_1hot,
                                                              reuse=False)

            def make_postwindow_lstm():
                cell = tf.nn.rnn_cell.LSTMCell(
                            PARAMS.lstm_size,
                            state_is_tuple=False,
                            # cell_clip=1,
                            initializer=tf.contrib.layers.xavier_initializer()
                            )

                return cell

            postwindowcells = [dropoutwrap(make_postwindow_lstm())
                               for _
                               in range(PARAMS.number_of_postwindow_layers)]

            all_lstm_layers = WindowedMultiRNNCell(dualcell, postwindowcells)

            lstm_zero_state = all_lstm_layers.zero_state(PARAMS.batch_size,
                                                         tf.float32)

            rnn_outputs, last_state = tf.nn.dynamic_rnn(
                            cell=alllayerscell,
                            inputs=input_placeholder,
                            initial_state=self.rnn_initial_state_placeholder,
                            dtype=tf.float32
                            )

        with tf.name_scope("FC_LAYER"):
            # There are skip connections to the final layer from
            # the input layer and all hidden layers except the window,
            # which is index=1 of the rnn_outputs.
            fclayer_inputs = ([self.input_placeholder, rnn_outputs[0]] +
                              rnn_outputs[2:])
            fclayer_input = array_ops.concat(fclayer_inputs, axis=1,
                                             name='fc_input')

            fc_output = tf.contrib.layers.fully_connected(
                            fclayer_input,
                            PARAMS.output_size,
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            )

            network_output = output_layer

        with tf.name_scope("LOSS"):
            # next_inputs contains the correct predicted point
            # from the training data. Split it into [x1, x2, eos] tensors.

            x1_data, x2_data, eos_data = (tf.squeeze(x, axis=2) for x in
                                          tf.split(2, 3,
                                                   self.next_inputs))

            # Frequency density of predicted points is compared to each
            # gaussian in the mix individually.
            # Easiest way to do this is to copy the input data
            # num_gaussians times, so that we can calculate frequency
            # densities for every gaussian at once.
            x1_data, x2_data = [tf.stack([x] * PARAMS.num_gaussians,
                                         axis=2) for x in [x1_data, x2_data]]

            # ___PROCESS THE OUTPUT_LAYER INTO THE GAUSSIAN MIXTURE___

            # > Take first element as bernoulli param for end-of-stroke.
            phat_bernoulli = output_layer[:, :, 0]
            self.p_bernoulli = tf.nn.sigmoid(phat_bernoulli)

            # > Split remaining elements into parameters for the gaussians.
            predicted_gaussian_params = tf.reshape(network_output[:, :, 1:],
                                                   [self.batch_size,
                                                    -1,
                                                    # sequence_len,
                                                    PARAMS.num_gaussians,
                                                    6])
            (phat_pi, phat_mu1, phat_mu2,
             phat_sigma1, phat_sigma2,
             phat_rho) = (tf.squeeze(x, axis=3) for x in
                          tf.split(3, 6, predicted_gaussian_params))

            # > Transform the phat (p-hat) parameters into proper distribution
            # parameters, by normalising them as described in
            # https://arxiv.org/pdf/1308.0850v5.pdf page 20.
            self.p_pi = tf.nn.softmax(1e-10 + phat_pi, name="self.p_pi")
            self.p_mu1 = phat_mu1
            self.p_mu2 = phat_mu2
            self.p_sigma1 = tf.exp(phat_sigma1)
            self.p_sigma2 = tf.exp(phat_sigma2)
            self.p_rho = tf.tanh(1e-10 + phat_rho)

            def density_2d_gaussian(x1, x2, mu1, mu2, s1, s2, rho):
                z = (tf.square(tf.div(tf.sub(x1, mu1), s1)) +
                     tf.square(tf.div(tf.sub(x2, mu2), s2)) -
                     2 * tf.div(tf.mul(rho,
                                       tf.mul(tf.sub(x1, mu1),
                                              tf.sub(x2, mu2))),
                                tf.mul(s1, s2)))
                R = 1 - tf.square(rho)
                numerator = tf.exp(tf.div(-z, 2 * R))
                normfactor = 2 * np.pi * tf.mul(tf.mul(s1, s2), tf.sqrt(R))
                result = tf.div(numerator, normfactor)
                return result

            # ___CALCULATE LOSS:
            #       > DENSITY IN THE CALCULATED GAUSSIAN MIX
            #       OF THE ACTUAL NEXT POINT
            #       > PROBABILITY OF END_OF_STROKE CHOOSING THE ACTUAL
            #       EOS VALUE FOR THE NEXT POINT
            #    (TAKE NEGATIVE LOG OF BOTH TO MAKE A MINIMISABLE LOSS)___

            # Positional loss:
            # Get the density of the actual next point in each gaussian,
            # then weight them by self.p_pi to get the weighted density.
            predicted_densities_per_gaussian = density_2d_gaussian(
                                                 x1_data, x2_data,
                                                 self.p_mu1, self.p_mu2,
                                                 self.p_sigma1, self.p_sigma2,
                                                 self.p_rho)
            weighted_densities = tf.mul(predicted_densities_per_gaussian, self.p_pi)
            density_by_timestep = tf.reduce_sum(weighted_densities, axis=2)

            # EOS loss:
            # standard cross-entropy loss.
            bernoulli_density = (self.p_bernoulli       * eos_data +
                                 (1 - self.p_bernoulli) * (1-eos_data))

            # Make these into losses
            loss_due_to_position = -tf.log(1e-10 + density_by_timestep)
            loss_due_to_stroke_end = -tf.log(1e-10 + bernoulli_density)
            loss_by_time_step = (loss_due_to_position + loss_due_to_stroke_end)
            self.total_loss = tf.reduce_mean(loss_by_time_step)

            tf.summary.scalar('sample_loss', self.total_loss)
            tf.summary.scalar('bernoulli_density', tf.reduce_mean(bernoulli_density))
            tf.summary.scalar('gaussian_density', tf.reduce_mean(density_by_timestep))
            # pi_var is a measure of how much the pi parameter varies across
            # time steps. This is an interesting measure of how well the network
            # is learning, so log it.
            pi_mean, pi_var = tf.nn.moments(self.p_pi, [1], name="pi_meanvar")
            tf.summary.scalar('pi_var', tf.reduce_mean(pi_var))

        with tf.name_scope("TRAIN"):

            rein_optimizer = tf.train.RMSPropOptimizer(self.lr_placeholder)
            self.global_step = tf.Variable( 0, name='global_step',
                                            trainable=False)

            # Calculate and clip the gradients.
            tvars = tf.trainable_variables()
            self.grads =  tf.gradients(self.total_loss, tvars)
            self.grads = [tf.clip_by_value(g, -PARAMS.grad_clip, PARAMS.grad_clip) for g in self.grads]

            # Apply the gradients.
            optimizer = tf.train.AdamOptimizer(self.lr_placeholder)
            self.reinforcement_train_op = optimizer.apply_gradients(zip(self.grads, tvars), global_step=self.global_step)
            self.summaries = tf.summary.merge_all()

    def kappa_zerostate(self):
        return np.zeros([self.batch_size, PARAMS.window_gaussians])

    def postwindowlstm_zerostate(self):
        return np.zeros([self.batch_size, PARAMS.lstm_size * 2])

    def lstm1_zerostate(self):
        return np.zeros([self.batch_size, PARAMS.lstm_size * 2])


if __name__ == "__main__":
    testmodel = HandwritingModel(generate_mode=False)
