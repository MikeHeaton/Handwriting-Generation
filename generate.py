import model
import matplotlib.pyplot as plt
import tensorflow as tf
from config import PARAMS
import numpy as np
from read_strokesets import Point, Stroke, StrokeSet
from tqdm import tqdm

def generate_sample(length=500, use_saved=True):
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
        prev_point = np.array([[[0.0, 0.0, 1]]])
        prev_state = np.zeros([1, 2*PARAMS.lstm_size*PARAMS.number_of_layers])

        print("Generating points...")
        all_offsets = []
        for _ in tqdm(range(length)):
            # Generate parameters for the next point distribution,
            # using the network.
            feed_dict = {generating_model.input_placeholder        : prev_point,
                         generating_model.initial_state_placeholder: prev_state}

            (bernoulli_param, pi, rho,
            mu1, mu2,
            sigma1, sigma2,
            prev_state) = sess.run([generating_model.p_bernoulli,
                                    generating_model.p_pi,
                                    generating_model.p_rho,
                                    generating_model.p_mu1,
                                    generating_model.p_mu2,
                                    generating_model.p_sigma1,
                                    generating_model.p_sigma2,
                                    generating_model.last_state],
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
    print("Scale parameters:")
    xmean, ymean, xsdev, ysdev = list(np.genfromtxt(PARAMS.samples_directory + "/" + PARAMS.data_scale_file))
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
            # print(newstroke.points)
            strokeset_strokes.append(Stroke(stroke_points))
            stroke_points = []
    newstrokeset = StrokeSet(strokeset_strokes)
    #print(newstrokeset.strokes)
    return StrokeSet(strokeset_strokes)

if __name__ == "__main__":
    all_offsets = generate_sample(length=500)

    print("Making strokeset")
    strokeset = strokeset_from_offsets(all_offsets)

    print("Plotting")
    strokeset.plot()
