#--- Parameters dictionary ---#
# Each file should have
# from config import PARAMS
# in header; then access parameters as
# PARAMS.foo.
class Params():
    # --- Neural Network params ------
    lstm_size = 400
    number_of_layers = 3
    sequence_len = 50
    learning_rate = 0.00001

    num_gaussians = 20
    output_size = 6 * num_gaussians + 1
    dropout_keep_prob = 0.8

    # --- Model training params ------
    samples_directory = "./training_data/"
    weights_directory = "./network_weights/"
    batch_size = 32

    num_epochs = 20
    use_saved = False
    eval_every = 100
    record_every = 10
    save_every = 100

    # --- Data reading params ---
    data_scale = 0.001







PARAMS = Params()
