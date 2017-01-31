#--- Parameters dictionary ---#
# Each file should have
# from config import PARAMS
# in header; then access parameters as
# PARAMS.foo.
import numpy as np

class Params():
    # --- Data reading params ---
    samples_directory = "./training_data"
    data_scale_file = "data_scale_params"

    # --- Neural Network params ------
    lstm_size = 400
    number_of_layers = 3
    sequence_len = 400

    num_gaussians = 20
    output_size = 6 * num_gaussians + 1
    dropout_keep_prob = 0.8

    grad_clip = 10

    # --- Model training params ------

    weights_directory = "./network_weights/"
    batch_size = 32

    learning_rate_init = 5e-3
    learning_rate_decay = 0.98

    num_epochs = 2000
    restrict_samples = None
    use_saved = False
    eval_every = 100
    record_every = 10
    save_every = 100








PARAMS = Params()
