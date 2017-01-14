import model
import matplotlib.pyplot as plt
import tensorflow as tf
from config import PARAMS
import numpy as np
from read_strokesets import Point, Stroke, StrokeSet
from tqdm import tqdm

def generate_sample(length=500, use_saved=True):
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

        def sample_from_gaussians(mu1, mu2, sigma1, sigma2, rho):
            # Sample from some number of 2d gaussians, given their parameters.
            # The covariance matrix is  [s1*s1      s1*s2*rho   ]
            #                           [s1*s2*rho  s2*s2       ]
            # , taking rho to be as defined in
            # http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc542.htm
            cov_matrix = np.array([[sigma1*sigma1,      sigma1*sigma2*rho],
                                   [sigma1*sigma2*rho,  sigma2*sigma2    ]])
            mean = np.array((mu1, mu2))
            return np.array([np.random.multivariate_normal(np.squeeze(mean[:,:,:,i]),
                                                           np.squeeze(cov_matrix[:,:,:,:,i]))
                             for i in range(PARAMS.num_gaussians)])

        prev_point = np.array([[[0.0, 0.0, 1]]])
        prev_state = np.zeros([1, 2*PARAMS.lstm_size*PARAMS.number_of_layers])

        print("Generating points...")
        all_offsets = []
        for _ in tqdm(range(length)):
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
            #print("bernoulli_param", bernoulli_param)
            print("pi", pi)
            #print("rho", rho)
            #print("mu1 mu2", mu1, mu2)
            #print("sigma1 sigma2", sigma1, sigma2)
            #print("Prev state", prev_state)
            gaussian_points = sample_from_gaussians(mu1, mu2,
                                                    sigma1, sigma2, rho)
            weighted_points = np.reshape(pi, [-1, 1]) * gaussian_points
            predicted_offset = np.sum(weighted_points, axis=0)

            end_stroke = np.random.binomial(1, bernoulli_param, size=None)

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
    for t in range(len(point_offsets)):
        point = Point(point.x + all_offsets[t,0],
                      point.y + all_offsets[t,1])
        #print(point.x, point.y)
        stroke_points.append(point)

        if all_offsets[t,2] == 1 or t == len(point_offsets) - 1:
            # When the third arg is 1, it's the end of a stroke.
            # print(stroke_points)
            newstroke = Stroke(stroke_points)
            # print(newstroke.points)
            strokeset_strokes.append(Stroke(stroke_points))
            stroke_points = []
    newstrokeset = StrokeSet(strokeset_strokes)
    #print(newstrokeset.strokes)
    return StrokeSet(strokeset_strokes)

if __name__ == "__main__":
    point_offsets = generate_sample(length=50)

    print("Making strokeset")
    strokeset = strokeset_from_offsets(point_offsets)

    print("Plotting")
    strokeset.plot()
