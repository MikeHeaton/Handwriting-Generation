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
        yield Sample(points_data, next_points_data)

class minibatches_from_samples:
    def __init__(self, preexisting_samples=[]):
        self.samples_to_use = preexisting_samples
    def __call__(self, list_of_samples):
        # Takes in a list of sample objects and adds them to any still in memory
        self.samples_to_use.extend(list_of_samples)

        # Shuffles them and puts them into minibatches
        random.shuffle(self.samples_to_use)
        for t in range(int(len(self.samples_to_use) / PARAMS.batch_size)):
            yield Minibatch(self.samples_to_use[t:t+PARAMS.batch_size])

        # The remainder becomes the list of samples still to be used,
        # to be added to the next call.
        self.samples_to_use = self.samples_to_use[
                                int(len(self.samples_to_use) / PARAMS.batch_size)
                                * PARAMS.batch_size
                                :]

def legacy_minibatch_generator_from_directory(dir, max_strokesets=None):
    # Takes a directory.
    # Reads all the strokesets, then yields it one at a time.

    print("Reading files from {} ...".format(dir))
    strokesets = read_strokesets.all_strokesets_from_dir(PARAMS.samples_directory, max_strokesets=max_strokesets)
    print("Done, read {} files.".format(len(strokesets)))
    """TODO: make all_strokesets_from_dir a generator as well for MORE SPEED
    / less boringness at the beginning. Need to figure out how to shuffle samples
    if this is done."""

    def minibatch_generator():
        for minibatch in minibatches_from_strokesets(strokesets):
            yield minibatch

    return minibatch_generator()

def minibatches_from_strokesets(strokesets_list):
    # Takes a list of strokesets_list
    # Shuffles the list, then reads the strokesets one at a time and yields
    # minibatches as it goes.
    random.shuffle(strokesets_list)

    # Initialise minibatch generator
    minibatch_generator = minibatches_from_samples()

    for strokeset in strokesets_list:
        print("Reading from new strokeset")
        strokeset_samples = training_samples_from_strokeset(strokeset)
        for minibatch in minibatch_generator(strokeset_samples):
            yield minibatch


"""-------------------------------------------------"""
"""-------------------------------------------------"""
"""-------------------------------------------------"""
"""-------------------------------------------------"""

def minibatch_generator_from_directory(dir, max_strokesets=None):
    print("Reading files from {} ...".format(dir))
    strokesets = read_strokesets.all_strokesets_from_dir(PARAMS.samples_directory, max_strokesets=max_strokesets)
    print("Done, read {} files.".format(len(strokesets)))

    random.shuffle(strokesets)
    strokesets.sort(key=lambda s:(len(s.to_numpy())-1)//PARAMS.sequence_len)

    for file_group in range( len(strokesets)//PARAMS.batch_size):
        yield v2_minibatches_from_list_of_strokesets(strokesets[file_group*PARAMS.batch_size: (file_group+1)*PARAMS.batch_size])


def v2_minibatches_from_list_of_strokesets(strokesets_list):
    # Takes a list of PARAMS.batch_size strokesets.
    # Makes as many minibatches as possible from them and returns a generator.
    def minibatch_generator():
        all_strokedata= [s.to_numpy() for s in strokesets_list]
        minibatches_in_sample = min([(len(s)-1)//PARAMS.sequence_len for s in all_strokedata])
        #print("{} MINIBATCHES IN THESE SAMPLES".format(minibatches_in_sample))
        for t in range(minibatches_in_sample):
            points_data = [s[t*PARAMS.sequence_len: (t+1)*PARAMS.sequence_len] for s in all_strokedata]
            next_points_data = [s[1+t*PARAMS.sequence_len: 1+(t+1)*PARAMS.sequence_len] for s in all_strokedata]
            yield Minibatch([Sample(points_data[i], next_points_data[i]) for i in range(len(points_data))])

    return minibatch_generator()

if __name__ == "__main__":
    training_data_generators = minibatch_generator_from_directory(
                                                    PARAMS.samples_directory
                                                    , max_strokesets=100
                                                    )
    for group in training_data_generators:
        print("NEW FILE GROUP")
        for x in group:
            print("POINTS\n",x.points_data.shape)
            print("NEXT POINTS\n",x.next_points_data.shape)
            print("--")
