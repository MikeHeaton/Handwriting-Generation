import model
import create_training_data
from config import PARAMS
import tensorflow as tf
from tqdm import tqdm
import os


with tf.Session() as sess:
    """
    Initialise objects and variables
    """

    # Initialise model and related objects
    print("Initialising model...")
    training_model = model.HandwritingModel()
    saver = tf.train.Saver()
    if PARAMS.use_saved:
        print("Restoring model weights...")
        saver.restore(sess, tf.train.latest_checkpoint(PARAMS.weights_directory))
    else:
        print("Initialising random weights...")
        sess.run(tf.global_variables_initializer())
    print("Done.")

    # Set train/Dev Summary directories and writers
    print("Setting up summary writers...")
    train_summary_directory = './summaries/train'
    train_summary_writer = tf.train.SummaryWriter(train_summary_directory, sess.graph)
    print("Done train summary")
    dev_summary_directory = './summaries/dev'
    dev_summary_writer = tf.train.SummaryWriter(dev_summary_directory, sess.graph)
    print("Done.")

    """
    Define the training and eval functions
    """

    def run_train_step(minibatch):
        feed_dict = {training_model.input_placeholder : minibatch.points_data,
                        training_model.next_inputs_placeholder : minibatch.next_points_data}

        """print("points ->\n", minibatch.points_data)
        print("next points ->\n", minibatch.next_points_data)
        stop = input()

        (bernoulli_param, pi, rho,
        mu1, mu2,
        sigma1, sigma2,
        prev_state) = sess.run([training_model.p_bernoulli,
                                training_model.p_pi,
                                training_model.p_rho,
                                training_model.p_mu1,
                                training_model.p_mu2,
                                training_model.p_sigma1,
                                training_model.p_sigma2,
                                training_model.last_state],
                                feed_dict = feed_dict)
        #print("bernoulli_param", bernoulli_param)
        print("pi", pi)
        #print("rho", rho)
        #print("mu1 mu2", mu1, mu2)
        #print("sigma1 sigma2", sigma1, sigma2)
        inp = input()
        #print("Prev state", prev_state)"""

        _, current_step, summary = sess.run([training_model.reinforcement_train_op,
                                            training_model.global_step,
                                            training_model.summaries],
                                            feed_dict = feed_dict)

        if current_step % PARAMS.record_every == 0:
            train_summary_writer.add_summary(summary, current_step)

    def run_all_dev(dev_set):
        print("""TODO: add evaluation""")

    def run_epoch(training_data):

        for minibatch in tqdm(training_data):
            run_train_step(minibatch)

            current_step = tf.train.global_step(sess, training_model.global_step)
            if current_step % PARAMS.eval_every == 0:
                pass
                """TODO: add eval"""
                #print("Evaluating at step {}:".format(current_step))
                #run_all_dev(None)
            if current_step % PARAMS.save_every == 0:
                saver.save(sess, PARAMS.weights_directory,
                            global_step=training_model.global_step)


    """
    Run training!
    """

    for epoch in range(PARAMS.num_epochs):
        print("---Beginning epoch {}---".format(epoch))

        """FETCH TRAINING DATA in minibatch form"""
        """TODO: figure out the best way of doing this. (Too many samples
        to fetch at once! ..?)"""
        print("Fetching training data...")
        training_data_generator = create_training_data.minibatch_generator_from_directory(
                                                        PARAMS.samples_directory
                                                        #, max_strokesets=10
                                                        )
        print("Training...")
        run_epoch(training_data_generator)

"""On the TODO list:
Implement evaluation / test (what's the correct word for it?)
"""
