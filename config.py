#--- Parameters dictionary ---#
# Each file should have
# from config import PARAMS
# in header; then access parameters as
# PARAMS.foo.
import numpy as np
import os
from collections import defaultdict

class Params():
    # --- Data reading params ---
    samples_directory = os.path.abspath("training_data")

    data_scale_file = "data_scale_params"

    # --- Neural Network params ------
    lstm_size = 256 # (was 400 for saved weights)
    number_of_postwindow_layers = 1
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

    num_epochs = 20
    restrict_samples = None
    use_saved = False
    eval_every = 100
    record_every = 10
    save_every = 100

    # --- Window params ------- #

    window_gaussians = 10
    max_char_len = 64

    int_to_char = {**{0: '.', 1: ' '},
                    **{i - 63: chr(i) for i in range(65,91)},
                    **{i - 69: chr(i) for i in range(97,123)}}
    #int_to_char = defaultdict(int, int_to_char)

    char_to_int = defaultdict(int, {c: i for i, c in int_to_char.items()})
    num_characters = len(int_to_char)


PARAMS = Params()

"""NOTES: consider cell_clip on the LSTM cell, it was used in StepI and worked.
Currently off because idk if it's necessary."""
