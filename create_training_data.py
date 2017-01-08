import read_strokesets
from config import PARAMS
import random
import numpy as np

class Sample():
    def __init__(self, points_data, next_points_data):
        self.points_data = points_data
        self.next_points_data = next_points_data

class Minibatch():
    def __init__(self, list_of_samples):
        self.points_data = np.array([x.points_data for x in list_of_samples])
        self.next_points_data = np.array([x.next_points_data for x in list_of_samples])

def training_samples_from_strokeset(strokeset):
    # Takes in a strokeset object
    # Returns all of the samples (length PARAMS.sequence_len)
    # from the strokeset. CURRENTLY we're returning all the sequences
    # of full length, ie those starting in positions up to len - sequence_len.
    """TODO: consider/test if this is the best method"""
    all_points = strokeset.to_numpy()
    samples_list = []

    for t in range(len(all_points) - PARAMS.sequence_len - 1):
        points_data = all_points[t: t + PARAMS.sequence_len]
        next_points_data = all_points[t + 1: t + 1 + PARAMS.sequence_len]
        samples_list.append(Sample(points_data, next_points_data))

    return samples_list

def minibatches_from_samples(list_of_samples):
    # Takes in a list of sample numpy arrays
    # Shuffles them and puts them into minibatches (ignoring remainder)
    random.shuffle(list_of_samples)
    print(len(list_of_samples), PARAMS.batch_size)
    for t in range(int(len(list_of_samples) / PARAMS.batch_size)):
        yield Minibatch(list_of_samples[t:t+PARAMS.batch_size])

def minibatches_from_directory(dir, max_strokesets=None):
    # Takes a directory.
    # Reads all the strokesets from them

    print("Reading files from {} ...".format(rootdir))
    strokesets = read_strokesets.all_strokesets_from_dir(PARAMS.samples_directory, max_strokesets=max_strokesets)
    print("Done, read {} files.".format(len(all_strokesets)))

    print("Creating samples from files...")
    samples = sum([training_samples_from_strokeset(ss) for ss in strokesets], [])
    print("All samples created; batching...")
    minibatches = minibatches_from_samples(samples)

    print("Done.")
    return minibatches

def minibatches_from_strokesets(strokesets_list):
    # Takes a list of strokesets_list
    # Shuffles the list, then reads the strokesets one at a time and yields
    # minibatches as it goes.
    random.shuffle(strokesets_list)
    preexisting_samples = []

    for strokeset in strokesets_list:


if __name__ == "__main__":
    print("Reading data from disk...")
    strokesets = read_strokesets.all_strokesets_from_dir(PARAMS.samples_directory, max_strokesets=None)
    print("Sampling & batching training data...")
    samples = sum([training_samples_from_strokeset(ss) for ss in strokesets], [])
    print(len(samples))
