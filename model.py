"""
model = tensorflow model

training routine:
    init model with params either random init or read from disk
    load all files from directory
    train on each one in model
    save the parameters down to disk
"""

from config import PARAMS
import tensorflow as tf
import numpy as np


class HandwritingModel:
    def __init__(self):

        with tf.name_scope("INPUT"):
            self.input_placeholder = tf.placeholder(tf.float32,
                                        shape=( PARAMS.batch_size,
                                                PARAMS.sequence_len,
                                                3))

        with tf.name_scope("LSTM_LAYERS"):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(
                            PARAMS.lstm_size,
                            state_is_tuple=True,
                            initializer=tf.contrib.layers.xavier_initializer()
                            )
            stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                            [lstm_cell] * PARAMS.number_of_layers,
                            state_is_tuple=True
                            )
            lstm_outputs, last_states = tf.nn.dynamic_rnn(
                            cell=stacked_lstm_cell,
                            dtype=tf.float32,
                            inputs=self.input_placeholder
                            )

        with tf.name_scope("FC_LAYER"):
            W = tf.get_variable("W_out",
                            shape=(PARAMS.lstm_size, PARAMS.output_size),
                            initializer=tf.contrib.layers.xavier_initializer()
                            )

            b = tf.get_variable("b_out",
                            shape=[PARAMS.output_size],
                            initializer=tf.contrib.layers.xavier_initializer()
                            )

            lstm_outputs = tf.reshape(lstm_outputs, [-1, PARAMS.lstm_size])
            output_layer =  tf.matmul(lstm_outputs, W)
            output_layer = tf.add(output_layer, b)
            output_layer = tf.reshape(output_layer, [PARAMS.batch_size,
                                                     PARAMS.sequence_len,
                                                     -1])
            """TODO: make this tensor multiplication into a pattern,
            to make it neater to reuse"""

            self.network_output = tf.sigmoid(output_layer,
                                            name="network_output")

        with tf.name_scope("LOSS"):
            # Read the actual points data for training and split.
            self.next_inputs_placeholder = tf.placeholder(tf.float32,
                                        shape=( PARAMS.batch_size,
                                                PARAMS.sequence_len,
                                                3))
            # > Copy the input data num_gaussians times, so that we can calculate
            #   frequency densities for every gaussian at once.

            self.x1_data, self.x2_data, self.eos_data = (tf.squeeze(x, axis=2) for x in tf.split(2, 3, self.next_inputs_placeholder))
            self.x1_data, self.x2_data = [tf.stack([x] * PARAMS.num_gaussians,
                                      axis=2) for x in [self.x1_data, self.x2_data]]

            # Take first element as bernoulli param for end-of-stroke.
            phat_bernoulli = output_layer[:, :, 0]
            p_bernoulli = tf.divide(1, 1 + tf.exp(phat_bernoulli))

            # Split remaining elements into parameters for the gaussians,
            # which are used to define the distribution mix for prediction.
            self.predicted_gaussian_params = tf.reshape(self.network_output[:, :, 1:],
                                                    [PARAMS.batch_size,
                                                     PARAMS.sequence_len,
                                                     PARAMS.num_gaussians,
                                                     6])

            phat_pi, phat_mu1, phat_mu2, phat_sigma1, phat_sigma2, phat_rho = (tf.squeeze(x, axis=3) for x in tf.split(3, 6, self.predicted_gaussian_params))

            # Transform the phat (p-hat) parameters into proper distribution
            # parameters, by normalising them as described in
            # https://arxiv.org/pdf/1308.0850v5.pdf page 20.
            p_pi = tf.nn.softmax(phat_pi, name="p_pi")
            p_mu1 = phat_mu1
            p_mu2 = phat_mu2
            p_sigma1 = tf.exp(phat_sigma1)
            p_sigma2 = tf.exp(phat_sigma2)
            p_rho = tf.tanh(phat_rho)

            def density_2d_gaussian(x1, x2, mu1, mu2, sigma1, sigma2, rho):
                Z = (tf.divide(tf.square(x1 - mu1), tf.square(sigma1)) +
                    tf.divide(tf.square(x2 - mu2), tf.square(sigma2)) -
                    tf.divide(2 * rho * (x1 - mu1) * (x1 - mu2), sigma1 * sigma2))

                R = 1-tf.square(rho)
                exponential = tf.exp(tf.divide(-Z, 2*R))
                density = tf.divide(exponential,
                                    2 * np.pi * sigma1 * sigma2 * tf.sqrt(R))

                return density

            self.predicted_densities_by_gaussian = density_2d_gaussian(self.x1_data, self.x2_data,
                                                        p_mu1, p_mu2,
                                                        p_sigma1, p_sigma2,
                                                        p_rho)

            # Weight densities_by_gaussian by p_pi to get prob density for
            # each time step.
            self.weighted_densities = tf.mul(self.predicted_densities_by_gaussian, p_pi)
            self.total_densities = tf.reduce_sum(self.weighted_densities, axis=2)
            self.loss_due_to_gaussians = -tf.log(self.total_densities) #<---- CN DELETE AFTER DEBUG
            self.loss_due_to_bernoulli = -tf.log(p_bernoulli + 2 * tf.mul(self.eos_data, p_bernoulli))
            self.loss_by_time_step = (-tf.log(self.total_densities) -
                            tf.log(p_bernoulli + 2 * tf.mul(self.eos_data, p_bernoulli))
                            )
            self.total_loss = tf.reduce_sum(self.loss_by_time_step)

        with tf.name_scope("TRAIN"):
            rein_optimizer = tf.train.RMSPropOptimizer(PARAMS.learning_rate)
            self.global_step = tf.Variable( 0, name='global_step',
                                            trainable=False)
            self.reinforcement_train_op = rein_optimizer.minimize(
                                            self.total_loss,
                                            global_step=self.global_step)
            """TODO: add training params: dropout?, grad clipping"""

if __name__ == "__main__":
    testmodel = HandwritingModel()
