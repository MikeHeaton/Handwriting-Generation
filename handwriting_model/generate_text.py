import model
import matplotlib.pyplot as plt
import tensorflow as tf
from config import PARAMS
import numpy as np
import read_strokesets
from read_strokesets import Point, Stroke, StrokeSet
from tqdm import tqdm
import sys
import os

def generate_sample(text, length=500, use_saved=True):
    # Generate a sequence of (normalised) offsets from the neural network.
    # Return it as an array.
    with tf.Session() as sess:
        print("Creating model...")
        generating_model = model.HandwritingModel(generate_mode=True)

        if use_saved:
            print("Loading variables...")
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(PARAMS.weights_directory))
        else:
            print("Initialising random variables...")
            sess.run(tf.initialize_all_variables())
        print("Done.")

        def sample_from_gaussian(mu1, mu2, sigma1, sigma2, rho):
            # Sample from a 2d gaussian, given the parameters.
            # The covariance matrix is  [s1*s1      s1*s2*rho   ]
            #                           [s1*s2*rho  s2*s2       ]
            # , taking rho to be as defined in
            # http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc542.htm
            cov_matrix = np.array([[sigma1*sigma1,      sigma1*sigma2*rho],
                                   [sigma1*sigma2*rho,  sigma2*sigma2    ]])
            mean = np.array((mu1, mu2))
            return np.random.multivariate_normal(mean, cov_matrix)

        # Initialise feed values for the network
        character_codes = [PARAMS.char_to_int[c] for c in text]
        text_data = np.zeros([1, PARAMS.max_char_len])
        text_data[0, 0: len(character_codes)] = character_codes
        input_len = np.ones([1], dtype=np.int32)

        text_length = np.zeros([1], dtype=np.int32)
        text_length[0] = len(text)

        prev_point = np.array([[[0.0, 0.0, 1]]])
        prev_l1_state = generating_model.lstm1_zerostate()
        prev_postwindow_state = generating_model.postwindowlstm_zerostate()
        prev_kappa = generating_model.kappa_zerostate()

        print("Generating points...")
        all_offsets = []
        for _ in tqdm(range(length)):
            feed_dict = {generating_model.input_placeholder                   : prev_point,
                         generating_model.inputs_length_placeholder           : input_len,
                         generating_model.l1_initial_state_placeholder        : prev_l1_state,
                         generating_model.postwindow_initial_state_placeholder: prev_postwindow_state,
                         generating_model.kappa_initial_placeholder           : prev_kappa,
                         generating_model.character_codes_placeholder         : text_data,
                         generating_model.character_lengths_placeholder       : text_length}

            (bernoulli_param, pi, rho,
            mu1, mu2,
            sigma1, sigma2,
            prev_l1_state,
            prev_postwindow_state,
            prev_kappa) = sess.run([generating_model.p_bernoulli,
                                    generating_model.p_pi,
                                    generating_model.p_rho,
                                    generating_model.p_mu1,
                                    generating_model.p_mu2,
                                    generating_model.p_sigma1,
                                    generating_model.p_sigma2,
                                    generating_model.final_l1_state,
                                    generating_model.last_postwindow_lstm_state,
                                    generating_model.final_kappa],
                                    feed_dict = feed_dict)

            # Using parameters, generate a randomly chosen next point.
            # > Choose end_stroke=1 with probability bernoulli_param
            end_stroke = np.random.binomial(1, bernoulli_param, size=None)

            # > Choose from among the gaussian distributions with
            # a distribution determined by pi, then sample from the chosen
            # gaussian distribution
            pi = np.reshape(pi, [-1])
            gaussian_choice = np.random.choice(list(range(len(pi))), p=pi)
            predicted_offset = sample_from_gaussian(mu1[0,0, gaussian_choice],
                                                    mu2[0,0, gaussian_choice],
                                                    sigma1[0,0, gaussian_choice],
                                                    sigma2[0,0, gaussian_choice],
                                                    rho[0,0, gaussian_choice])

            prev_point = np.reshape(np.append(predicted_offset, end_stroke), [1, 1, -1])
            all_offsets.append(np.squeeze(prev_point))

        return np.array(all_offsets)

def strokeset_from_offsets(all_offsets):
    # Takes a list of point offsets
    # Turns them into a strokeset & returns it
    print(all_offsets)
    strokeset_strokes = []
    point=Point(0,0)
    stroke_points = [point]

    # Offsets have been normalised to mean 0, stdev 1.
    # We can de-normalise them.
    scale = read_strokesets.get_data_scale(
                        os.path.join(PARAMS.samples_directory, "strokes_data"))
    xmean, ymean, xsdev, ysdev = list(scale)
    print(xmean, ymean, xsdev, ysdev)

    for t in range(len(all_offsets)):
        xoffset_normed = all_offsets[t,0]
        yoffset_normed = all_offsets[t,1]
        xoffset_raw = xoffset_normed * xsdev + xmean
        yoffset_raw = yoffset_normed * ysdev + ymean

        point = Point(point.x + xoffset_raw,
                      point.y + yoffset_raw)
        stroke_points.append(point)

        if all_offsets[t,2] == 1 or t == len(all_offsets) - 1:
            # When the third arg is 1, it's the end of a stroke.
            newstroke = Stroke(stroke_points)
            strokeset_strokes.append(Stroke(stroke_points))
            stroke_points = []
    newstrokeset = StrokeSet(strokeset_strokes)
    return StrokeSet(strokeset_strokes)

if __name__ == "__main__":
    all_offsets = generate_sample(sys.argv[1], length=500, use_saved=True)

    print("Making strokeset")
    strokeset = strokeset_from_offsets(all_offsets)

    print("Plotting")
    strokeset.plot()
