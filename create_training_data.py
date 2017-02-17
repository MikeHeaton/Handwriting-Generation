import read_strokesets
from config import PARAMS
import random
import numpy as np
import generate
import preprocess_data

class Sample():
    def __init__(self, offsets_data):
        # Create point & next_point lists from the input data.
        # A sample contains
        self.offsets_data = offsets_data[:-1]
        self.next_offsets_data = offsets_data[1:]

class Minibatch():
    def __init__(self, list_of_samples):
        # A minibatch is designed to contain PARAMS.batch_size input feeds
        # for the network.
        self.offsets_data = np.array([x.offsets_data for x in list_of_samples])
        self.next_offsets_data = np.array([x.next_offsets_data for x in list_of_samples])

def minibatch_generator_from_directory(dir, max_strokesets=None):
    # Input: a relative directory path
    # Returns: minibatches created from data from the directory
    print("Reading files from {} ...".format(dir))
    strokesets = read_strokesets.all_strokesets_from_dir(PARAMS.samples_directory, max_strokesets=max_strokesets, use_scale=True)
    print("Done, read {} files.".format(len(strokesets)))

    print(preprocess_data.meanstd(strokesets))
    # To randomise the order of the samples, shuffle them.
    # Then sort by length so that batches of similar length are grouped together
    # (because ends of groups are hacked off)
    random.shuffle(strokesets)
    strokesets.sort(key=lambda s:(len(s.to_numpy())-1)//PARAMS.sequence_len)

    for file_group in range( len(strokesets)//PARAMS.batch_size):
        yield minibatches_from_list_of_strokesets(strokesets[file_group*PARAMS.batch_size: (file_group+1)*PARAMS.batch_size])


def minibatches_from_list_of_strokesets(strokesets_list):
    # Takes a list of PARAMS.batch_size many strokesets.
    # Makes as many minibatches as possible from them and returns a generator.
    def minibatch_generator():
        all_offsetdata= [s.to_numpy() for s in strokesets_list]
        minibatches_in_sample = min([(len(s)-1)//PARAMS.sequence_len for s in all_offsetdata])
        #print("{} MINIBATCHES IN THESE SAMPLES".format(minibatches_in_sample))
        for t in range(minibatches_in_sample):
            offsets_data = [s[t*PARAMS.sequence_len: 1+(t+1)*PARAMS.sequence_len] for s in all_offsetdata]
            # next_offsets_data = [s[1+t*PARAMS.sequence_len: 1+(t+1)*PARAMS.sequence_len] for s in all_offsetdata]
            yield Minibatch([Sample(offsets_data[i])for i in range(len(offsets_data))])
            # , next_offsets_data[i])

    return minibatch_generator()

if __name__ == "__main__":
    training_data_generators = minibatch_generator_from_directory(
                                                    PARAMS.samples_directory
                                                    #, max_strokesets=100
                                                    )
    for group in training_data_generators:
        print("NEW FILE GROUP")
        for x in group:
            print("MINIBATCH")
            print("OFFSETS\n",x.offsets_data.shape)
            print("NEXT OFFSETS\n",x.next_offsets_data.shape)
            generate.strokeset_from_offsets(x.offsets_data[0,:,:]).plot()
            print("--")
