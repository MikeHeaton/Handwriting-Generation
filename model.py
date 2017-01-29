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
    def __init__(self, generate_mode=False):
        self.generate_mode = generate_mode
        if self.generate_mode:
            batch_size = 1
            sequence_len = 1
        else:
            batch_size = PARAMS.batch_size
            sequence_len = PARAMS.sequence_len

        with tf.name_scope("INPUT"):
            self.input_placeholder = tf.placeholder(tf.float32,
                                        shape=( batch_size,
                                                sequence_len,
                                                3))

        with tf.name_scope("LSTM_LAYERS"):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(
                            PARAMS.lstm_size,
                            state_is_tuple=False,
                            cell_clip=1,
                            initializer=tf.contrib.layers.xavier_initializer()
                            )
            stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                            [lstm_cell] * PARAMS.number_of_layers,
                            state_is_tuple=False
                            )

            if (not self.generate_mode) and PARAMS.dropout_keep_prob < 1:
                stacked_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(stacked_lstm_cell, output_keep_prob = PARAMS.dropout_keep_prob)

            self.lstm_zero_state = stacked_lstm_cell.zero_state(
                                                    batch_size=batch_size,
                                                    dtype=tf.float32)

            self.initial_state_placeholder = tf.placeholder(
                            dtype=tf.float32,
                            shape=[batch_size, PARAMS.lstm_size*2*PARAMS.number_of_layers]
                            )

            # Dynamic_rnn implementation
            lstm_outputs, self.last_state = tf.nn.dynamic_rnn(
                            cell=stacked_lstm_cell,
                            inputs=self.input_placeholder,
                            initial_state=self.initial_state_placeholder,
                            dtype=tf.float32
                            )

            """
            # Seq2seq implementation
            inputs = tf.split(1, sequence_len, self.input_placeholder)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

            lstm_outputs, self.last_state = tf.nn.seq2seq.rnn_decoder(
                                                    inputs,
                                                    initial_state=self.initial_state_placeholder,
                                                    cell=stacked_lstm_cell,
                                                    loop_function=None)"""


            #lstm_outputs = tf.reshape(tf.concat(1, lstm_outputs), [batch_size, sequence_len, PARAMS.lstm_size])

        with tf.name_scope("FC_LAYER"):
            W = tf.get_variable("W_out",
                            shape=(PARAMS.lstm_size, PARAMS.output_size),
                            initializer= tf.contrib.layers.xavier_initializer()
                            )

            b = tf.get_variable("b_out",
                            #shape=[PARAMS.output_size],
                            initializer= tf.zeros_initializer(shape=[PARAMS.output_size])
                            )

            lstm_outputs = tf.reshape(lstm_outputs, [-1, PARAMS.lstm_size])

            output_layer = tf.matmul(lstm_outputs, W)
            output_layer = tf.add(output_layer, b)
            output_layer = tf.reshape(output_layer, [batch_size,
                                                     sequence_len,
                                                     -1])

            """TODO: make this tensor multiplication into a pattern,
            to make it neater to reuse"""

            network_output = output_layer

        with tf.name_scope("LOSS"):
            # Read the actual points data for training and split.
            self.next_inputs_placeholder = tf.placeholder(tf.float32,
                                        shape=( batch_size,
                                                sequence_len,
                                                3))
            # > Copy the input data num_gaussians times, so that we can calculate
            #   frequency densities for every gaussian at once.

            x1_data, x2_data, eos_data = (tf.squeeze(x, axis=2) for x in tf.split(2, 3, self.next_inputs_placeholder))
            x1_data, x2_data = [tf.stack([x] * PARAMS.num_gaussians,
                                      axis=2) for x in [x1_data, x2_data]]

            # Take first element as bernoulli param for end-of-stroke.
            phat_bernoulli = output_layer[:, :, 0]
            self.p_bernoulli = tf.nn.sigmoid(phat_bernoulli)

            # Split remaining elements into parameters for the gaussians,
            # which are used to define the distribution mix for prediction.
            predicted_gaussian_params = tf.reshape(network_output[:, :, 1:],
                                                    [batch_size,
                                                     sequence_len,
                                                     PARAMS.num_gaussians,
                                                     6])

            #predicted_gaussian_params = tf.check_numerics(predicted_gaussian_params, "predicted_gaussian_params", name=None)

            phat_pi, phat_mu1, phat_mu2, phat_sigma1, phat_sigma2, phat_rho = (tf.squeeze(x, axis=3) for x in tf.split(3, 6, predicted_gaussian_params))

            # Transform the phat (p-hat) parameters into proper distribution
            # parameters, by normalising them as described in
            # https://arxiv.org/pdf/1308.0850v5.pdf page 20.
            self.p_pi = tf.nn.softmax(1e-10 + phat_pi, name="self.p_pi")
            self.p_mu1 = phat_mu1
            self.p_mu2 = phat_mu2
            self.p_sigma1 = tf.exp(phat_sigma1)
            self.p_sigma2 = tf.exp(phat_sigma2)
            self.p_rho = tf.tanh(1e-10 + phat_rho)


            def density_2d_gaussian(x1, x2, mu1, mu2, sigma1, sigma2, rho):
                Z = (tf.divide(tf.square(x1 - mu1), 1e-10 + tf.square(sigma1)) +
                    tf.divide(tf.square(x2 - mu2), 1e-10 + tf.square(sigma2)) -
                    tf.divide(2 * rho * (x1 - mu1) * (x1 - mu2), 1e-10 + sigma1 * sigma2))

                R = 1-tf.square(rho)
                exponential = tf.exp(tf.divide(-Z, 1e-10 + 2*R))
                density = tf.divide(exponential,
                                    1e-10 + 2 * np.pi * sigma1 * sigma2 * tf.sqrt(1e-4 + R))

                return density

            predicted_densities_by_gaussian = density_2d_gaussian(x1_data, x2_data,
                                                        self.p_mu1, self.p_mu2,
                                                        self.p_sigma1, self.p_sigma2,
                                                        self.p_rho)


            # Weight densities_by_gaussian by self.p_pi to get prob density for
            # each time step.
            weighted_densities = tf.mul(predicted_densities_by_gaussian, self.p_pi)
            density_by_timestep = tf.reduce_sum(weighted_densities, axis=2)
            print(density_by_timestep)
            loss_due_to_gaussians = -tf.log(1e-10 + density_by_timestep)
            #loss_due_to_gaussians = tf.Print(loss_due_to_gaussians, [self.p_sigma1])
            loss_due_to_bernoulli = -tf.log(1e-10 + self.p_bernoulli * eos_data +
                                            (1 - self.p_bernoulli) * (1-eos_data))
            loss_by_time_step = (loss_due_to_gaussians + loss_due_to_bernoulli)
            self.total_loss = tf.reduce_mean(loss_by_time_step)
            tf.summary.scalar('sample_loss', self.total_loss)
            # self.total_loss = tf.check_numerics(tf.Print(self.total_loss, [self.total_loss]), "totalloss")

        with tf.name_scope("TRAIN"):
            self.lr_placeholder = tf.placeholder(tf.float32, name='learning_rate')
            rein_optimizer = tf.train.RMSPropOptimizer(self.lr_placeholder)
            self.global_step = tf.Variable( 0, name='global_step',
                                            trainable=False)

            tvars = tf.trainable_variables()
            print(tvars)
            self.grads =  tf.gradients(self.total_loss, tvars)

            """GRADIENT CLIPPING"""
            self.grads = [tf.clip_by_value(g, -PARAMS.grad_clip, PARAMS.grad_clip) for g in self.grads]
            # self.grads = [tf.Print(g, [g]) for g in self.grads]

            #optimizer = tf.train.AdamOptimizer(self.lr_placeholder)
            optimizer = tf.train.RMSPropOptimizer(self.lr_placeholder)

            self.reinforcement_train_op = optimizer.apply_gradients(zip(self.grads, tvars))

            self.summaries = tf.summary.merge_all()

if __name__ == "__main__":
    testmodel = HandwritingModel(generate_mode=True)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(PARAMS.weights_directory))
        print(testmodel.generate_sample(sess))
