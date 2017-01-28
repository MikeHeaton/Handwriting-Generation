import read_strokesets
from config import PARAMS
import numpy as np

def load_files(dir):
    # Input: a relative directory path

    # Reads all strokesets from the directory;
    # figures out the scaling factors to apply to normalise
    # all the values in the strokesets to mean 0, stdev 1

    # Returns: minibatches created from data from the directory
    print("Reading files from {} ...".format(dir))
    strokesets = read_strokesets.all_strokesets_from_dir(PARAMS.samples_directory, max_strokesets=None, use_scale=False)
    print("Done, read {} files.".format(len(strokesets)))
    return strokesets

def meanstd(strokesets):
    alldata = np.concatenate([ss.to_numpy() for ss in strokesets], axis=0)[:,[0,1]]
    mean = np.mean(alldata, axis=0)
    std = np.std(alldata, axis=0)

    print("MEAN OFFSET: ", mean)
    print("STD OFFSET: ", std)
    return mean, std

if __name__ == "__main__":
    strokesets = load_files(PARAMS.samples_directory)
    mean, std = meanstd(strokesets)
    np.savetxt(PARAMS.samples_directory + "/" + PARAMS.data_scale_file, np.concatenate((mean, std), axis=0))
    print(np.genfromtxt(PARAMS.samples_directory + "/" + PARAMS.data_scale_file))
