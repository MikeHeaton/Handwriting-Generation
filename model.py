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
                            initializer=tf.contrib.layers.xavier_initializer()
                            )
            self.stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                            [lstm_cell] * PARAMS.number_of_layers,
                            state_is_tuple=False
                            )

            """DO THIS THING:
            http://stackoverflow.com/questions/38241410/tensorflow-remember-lstm-state-for-next-batch-stateful-lstm"""
            self.lstm_zero_state = self.stacked_lstm_cell.zero_state(
                                                    batch_size=batch_size,
                                                    dtype=tf.float32)

            print(self.lstm_zero_state)


            """np.zeros([PARAMS.number_of_layers, 2,
                                           PARAMS.batch_size, PARAMS.lstm_size],
                                           dtype=np.float32)"""


            #self.state_placeholder = tf.placeholder(tf.float32, [PARAMS.number_of_layers, 2, PARAMS.batch_size, PARAMS.lstm_size])
            #print(self.lstm_zero_state)
            self.initial_state_placeholder = tf.placeholder_with_default(
                            self.lstm_zero_state,
                            shape=[batch_size, PARAMS.lstm_size*2*PARAMS.number_of_layers]
                            )

            """l = tf.unpack(self.initial_state_placeholder, axis=0)
            print(self.stacked_lstm_cell.state_size, "<-----")
            rnn_tuple_state = tuple(
                                #[tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],
                                [(l[idx][0], l[idx][1])
                                 for idx in range(PARAMS.number_of_layers)]
                                )
            print(rnn_tuple_state)"""
            lstm_outputs, self.last_state = tf.nn.dynamic_rnn(
                            cell=self.stacked_lstm_cell,
                            inputs=self.input_placeholder,
                            initial_state=self.initial_state_placeholder,
                            dtype=tf.float32
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
            output_layer = tf.matmul(lstm_outputs, W)
            output_layer = tf.add(output_layer, b)
            output_layer = tf.reshape(output_layer, [batch_size,
                                                     sequence_len,
                                                     -1])
            """TODO: make this tensor multiplication into a pattern,
            to make it neater to reuse"""

            self.network_output = tf.sigmoid(output_layer,
                                            name="network_output")

        with tf.name_scope("LOSS"):
            # Read the actual points data for training and split.
            self.next_inputs_placeholder = tf.placeholder(tf.float32,
                                        shape=( batch_size,
                                                sequence_len,
                                                3))
            # > Copy the input data num_gaussians times, so that we can calculate
            #   frequency densities for every gaussian at once.

            self.x1_data, self.x2_data, self.eos_data = (tf.squeeze(x, axis=2) for x in tf.split(2, 3, self.next_inputs_placeholder))
            self.x1_data, self.x2_data = [tf.stack([x] * PARAMS.num_gaussians,
                                      axis=2) for x in [self.x1_data, self.x2_data]]

            # Take first element as bernoulli param for end-of-stroke.
            phat_bernoulli = output_layer[:, :, 0]
            self.p_bernoulli = tf.divide(1, 1 + tf.exp(phat_bernoulli))

            # Split remaining elements into parameters for the gaussians,
            # which are used to define the distribution mix for prediction.
            self.predicted_gaussian_params = tf.reshape(self.network_output[:, :, 1:],
                                                    [batch_size,
                                                     sequence_len,
                                                     PARAMS.num_gaussians,
                                                     6])

            phat_pi, phat_mu1, phat_mu2, phat_sigma1, phat_sigma2, phat_rho = (tf.squeeze(x, axis=3) for x in tf.split(3, 6, self.predicted_gaussian_params))

            # Transform the phat (p-hat) parameters into proper distribution
            # parameters, by normalising them as described in
            # https://arxiv.org/pdf/1308.0850v5.pdf page 20.
            self.p_pi = tf.nn.softmax(phat_pi, name="self.p_pi")
            self.p_mu1 = phat_mu1
            self.p_mu2 = phat_mu2
            self.p_sigma1 = tf.exp(phat_sigma1)
            self.p_sigma2 = tf.exp(phat_sigma2)
            self.p_rho = tf.tanh(phat_rho)

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
                                                        self.p_mu1, self.p_mu2,
                                                        self.p_sigma1, self.p_sigma2,
                                                        self.p_rho)

            # Weight densities_by_gaussian by self.p_pi to get prob density for
            # each time step.
            self.weighted_densities = tf.mul(self.predicted_densities_by_gaussian, self.p_pi)
            self.total_densities = tf.reduce_sum(self.weighted_densities, axis=2)
            loss_due_to_gaussians = -tf.log(self.total_densities)
            loss_due_to_bernoulli = -tf.log(1 - self.eos_data + (2*self.eos_data - 1) * self.p_bernoulli)
            self.loss_by_time_step = (loss_due_to_gaussians + loss_due_to_bernoulli)
            self.total_loss = tf.reduce_mean(self.loss_by_time_step)
            tf.summary.scalar('sample_loss', self.total_loss)

        with tf.name_scope("TRAIN"):
            rein_optimizer = tf.train.RMSPropOptimizer(PARAMS.learning_rate)
            self.global_step = tf.Variable( 0, name='global_step',
                                            trainable=False)
            self.reinforcement_train_op = rein_optimizer.minimize(
                                            self.total_loss,
                                            global_step=self.global_step)
            self.summaries = tf.summary.merge_all()

            """TODO: add training params: dropout?, grad clipping"""
    '''
    def generate_sample(self, sess, length=500):
        if not self.generate_mode:
            print("Model not created in generate mode.\n",
                    "Please recreate with generate_mode=True.")
            return 0

        def sample_from_gaussians(mu1, mu2, sigma1, sigma2, rho):
            # Sample from some number of 2d gaussians, given their parameters.
            # The covariance matrix is  [s1*s1      s1*s2*rho   ]
            #                           [s1*s2*rho  s2*s2       ]
            # taking to be the rho defined in
            # http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc542.htm
            cov_matrix = np.array([[sigma1*sigma1,      sigma1*sigma2*rho],
                                   [sigma1*sigma2*rho,  sigma2*sigma2    ]])
            mean = np.array((mu1, mu2))
            return np.array([np.random.multivariate_normal(np.squeeze(mean[:,:,:,i]),
                                                           np.squeeze(cov_matrix[:,:,:,:,i]))
                             for i in range(PARAMS.num_gaussians)])

        prev_point = np.array([[[0.0, 0.0, 1]]])
        prev_state = np.zeros([1, 2*PARAMS.lstm_size*PARAMS.number_of_layers])

        all_offsets = []
        for _ in range(length):
            feed_dict = {self.input_placeholder             : prev_point,
                         self.initial_state_placeholder     : prev_state}

            (bernoulli_param, pi, rho,
            mu1, mu2,
            sigma1, sigma2,
            prev_state) = sess.run( [self.p_bernoulli, self.p_pi, self.p_rho,
                                     self.p_mu1, self.p_mu2,
                                     self.p_sigma1, self.p_sigma2,
                                     self.last_state],
                                    feed_dict = feed_dict)
            gaussian_points = sample_from_gaussians(mu1, mu2,
                                                    sigma1, sigma2, rho)
            weighted_points = np.reshape(pi, [-1, 1]) * gaussian_points
            predicted_offset = np.sum(weighted_points, axis=0)

            end_stroke = np.random.binomial(1, bernoulli_param, size=None)

            prev_point = np.reshape(np.append(predicted_offset, 0), [1,1,-1])
            # prev_point = np.reshape(np.append(predicted_offset, end_stroke), [1,1,-1])
            all_offsets.append(np.squeeze(prev_point))
        return np.array(all_offsets)'''


if __name__ == "__main__":
    testmodel = HandwritingModel(generate_mode=True)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(PARAMS.weights_directory))
        #sess.run(tf.initialize_all_variables())
        print(testmodel.generate_sample(sess))
