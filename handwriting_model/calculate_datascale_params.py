import datapipeline
from config import PARAMS
import numpy as np
import pandas as pd

def load_files(dir):
    # Input: a relative directory path

    # Reads all strokesets from the directory;
    # figures out the scaling factors to apply to normalise
    # all the values in the strokesets to mean 0, stdev 1

    # Returns: minibatches created from data from the directory
    print("Reading files from {} ...".format(dir))
    samples = datapipeline.generate_samples_from_dir(PARAMS.samples_directory)
    strokesets = [s.strokeset for s in samples]
    print("Done, read {} files.".format(len(strokesets)))
    return strokesets

def meanstd(strokesets):
    # Calculates the mean and standard deviation of all of a set of strokesets.
    alldata = np.concatenate([ss.to_numpy() for ss in strokesets], axis=0)[:,[0,1]]
    mean = np.mean(alldata, axis=0)
    std = np.std(alldata, axis=0)

    print("MEAN OFFSET: ", mean)
    print("STD OFFSET: ", std)

    offsetdata = (alldata - mean) / std
    return mean, std

def length_statistics(strokesets):
    alldata = pd.Series([len(ss.to_numpy() )for ss in strokesets])
    print(alldata.describe())
    print([alldata.quantile(i/20) for i in range(21)])
    return alldata

if __name__ == "__main__":
    strokesets = load_files(PARAMS.samples_directory)

    length_statistics(strokesets)
    mean, std = meanstd(strokesets)
    np.savetxt(PARAMS.data_scale_file, np.concatenate((mean, std), axis=0))
    print(np.genfromtxt(PARAMS.samples_directory + "/" + PARAMS.data_scale_file))
