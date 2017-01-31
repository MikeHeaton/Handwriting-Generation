import model
import create_training_data
from config import PARAMS
import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np
from tensorflow.python import debug as tf_debug

with tf.Session() as sess:
    """
    Initialise objects and variables
    """

    print("Initialising model...")
    training_model = model.HandwritingModel()
    saver = tf.train.Saver()
    if PARAMS.use_saved:
        print("Restoring model weights...")
        saver.restore(sess, tf.train.latest_checkpoint(PARAMS.weights_directory))
    else:
        print("Initialising random weights...")
        sess.run(tf.global_variables_initializer())
        saver.save(sess, PARAMS.weights_directory,
                    global_step=training_model.global_step)
    print("Done.")

    print("Setting up summary writers...")
    train_summary_directory = './summaries/train'
    train_summary_writer = tf.summary.FileWriter(train_summary_directory, sess.graph)
    print("Created train summary")
    dev_summary_directory = './summaries/dev'
    dev_summary_writer = tf.summary.FileWriter(dev_summary_directory, sess.graph)
    print("Created dev summary.")

    """
    Define the training and eval functions
    """

    def run_train_step(minibatch, lr, init_state=None):
        if init_state is None:
            init_state = training_model.lstm_zero_state.eval()

        """print("-----RUN TRAIN STEP-----\nOFFSETS")
        print(minibatch.offsets_data)
        print(minibatch.offsets_data.mean(axis=1))
        print("TARGETS")
        print(minibatch.next_offsets_data)
        print(minibatch.next_offsets_data.mean(axis=1))"""

        feed_dict = {training_model.input_placeholder : minibatch.offsets_data,
                        training_model.next_inputs_placeholder : minibatch.next_offsets_data,
                        training_model.initial_state_placeholder : init_state,
                        training_model.lr_placeholder: lr}

        _, current_step, summary, last_state = sess.run([training_model.reinforcement_train_op,
                                            training_model.global_step,
                                            training_model.summaries,
                                            training_model.last_state],
                                            feed_dict = feed_dict)

        if current_step % PARAMS.record_every == 0:
            train_summary_writer.add_summary(summary, current_step)

        return last_state, lr, current_step

    def run_all_dev(dev_set):
        print("""TODO: add evaluation""")

    def run_epoch(training_data, lr):

        for file_group in tqdm(training_data):
            init_state = None

            for minibatch in file_group:
                _, lr, current_step = run_train_step(minibatch, lr, init_state)
                # init_state = run_train_step(minibatch, lr, init_state)
                print("CURRENT STEP:", current_step)

                if current_step % PARAMS.eval_every == 0:
                    pass
                    """TODO: add eval"""
                    #print("Evaluating at step {}:".format(current_step))
                    #run_all_dev(None)
                if current_step % PARAMS.save_every == 0:
                    print("SAVING")
                    saver.save(sess, PARAMS.weights_directory,
                                global_step=training_model.global_step)
        return lr

    """
    Run training!
    """
    lr = PARAMS.learning_rate_init

    for epoch in range(PARAMS.num_epochs):
        print("---Beginning epoch {}---".format(epoch))
        print("Learning rate = {0:f}".format(lr))

        """FETCH TRAINING DATA in minibatch form"""
        print("Fetching training data...")
        training_data_generator = create_training_data.minibatch_generator_from_directory(
                                        PARAMS.samples_directory,
                                        max_strokesets=PARAMS.restrict_samples
                                        )
        print("Training...")
        lr = run_epoch(training_data_generator, lr)
        lr = lr * PARAMS.learning_rate_decay
