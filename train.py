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
    train_summary_directory = './summaries/train'
    train_summary_writer = tf.train.SummaryWriter(train_summary_directory, sess.graph)
    dev_summary_directory = './summaries/dev'
    dev_summary_writer = tf.train.SummaryWriter(dev_summary_directory, sess.graph)



    """
    Define the training and eval functions
    """

    def run_train_step(minibatch):
        feed_dict = {training_model.input_placeholder : minibatch.points_data,
                        training_model.next_inputs_placeholder : minibatch.next_points_data}

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
                                                        # , max_strokesets=10)
                                                        )
        print("Training...")
        run_epoch(training_data_generator)

"""On the list:

Implement evaluation / test (what's the correct word for it?)
"""
