import tensorflow as tf
from tqdm import tqdm

import txtgen_model as model
import datapipeline
from config import PARAMS


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

    def run_train_step(minibatch, lr, state_info=None):
        if state_info is None:
            state_info = {"l1_init_state": training_model.lstm_1_zero_state,
                            "postwindow_init_state": training_model.postwindow_lstm_zero_state,
                            "kappa_init_state": training_model.kappa_zero_state}

        feed_dict = {
        training_model.input_placeholder: minibatch.inputs_data,
        training_model.next_inputs_placeholder: minibatch.outputs_data,
        training_model.inputs_length_placeholder: minibatch.sequence_lengths,
            training_model.l1_initial_state_placeholder: state_info["l1_init_state"],
        training_model.postwindow_initial_state_placeholder: state_info["postwindow_init_state"],
        training_model.kappa_initial_placeholder: state_info["kappa_init_state"],
        training_model.lr_placeholder: lr,
        training_model.character_codes_placeholder: minibatch.text_data,
        training_model.character_lengths_placeholder: minibatch.text_lengths}

        (_,
        current_step,
        summary,
        l1_final_state,
        postwindow_final_state,
        kappa_final_state) = sess.run([training_model.reinforcement_train_op,
                                        training_model.global_step,
                                        training_model.summaries,
                                        training_model.final_l1_state,
                                        training_model.last_postwindow_lstm_state,
                                        training_model.final_kappa],
                                        feed_dict = feed_dict)

        if current_step % PARAMS.record_every == 0:
            train_summary_writer.add_summary(summary, current_step)

        return {"l1_init_state": l1_final_state,
                "postwindow_init_state": postwindow_final_state,
                "kappa_init_state": kappa_final_state}, current_step

    def run_all_dev(dev_set):
        print("""TODO: add evaluation""")

    def run_epoch(training_data_generator, lr):

        for minibatch in tqdm(training_data_generator):
            state_info = None
            state_info, current_step = run_train_step(minibatch, lr, state_info)
            #print("CURRENT STEP:", current_step)

            if current_step % PARAMS.eval_every == 0:
                pass
                """TODO: add eval"""
                #print("Evaluating at step {}:".format(current_step))
                #run_all_dev(None)
            if current_step % PARAMS.save_every == 0:
                #print("SAVING")
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
        training_data_generator = datapipeline.generate_minibatches_from_dir(PARAMS.samples_directory)
        print("Training...")
        lr = run_epoch(training_data_generator, lr)
        lr = lr * PARAMS.learning_rate_decay
