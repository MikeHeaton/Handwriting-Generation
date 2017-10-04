import read_chardata
import read_strokesets
import os
from config import PARAMS
import re
from collections import defaultdict
import numpy as np
import queue

class Sample:
    def __init__(self, strokeset, coded_text, name, linenum):
        self.strokeset = strokeset
        self.text = coded_text

        # ID information
        self.name = name
        self.line = linenum

class Minibatch():
    def __init__(self, offsets_data, text_data,
                    sequence_lengths, text_lengths):
        # A minibatch is a container class for PARAMS.batch_size input feeds
        # for the network.
        self.inputs_data = offsets_data[:, :-1, :]
        self.outputs_data = offsets_data[:, 1:, :]
        self.text_data = text_data
        self.sequence_lengths = sequence_lengths
        self.text_lengths = text_lengths

class BatchBucketQueue():
    # Items are pushed onto the queue using the push function.
    # When a bucket is full, the items are pushed to a top_queue.
    # Pull returns the first element from the top_queue.
    # Result: minibatches grouped by the bucketing_function.
    def __init__(self, bucketing_function, batch_size):
        self.buckets = defaultdict(list)
        self.topqueue = queue.Queue()
        self.batch_size = batch_size
        self.bucketing_function = bucketing_function

    def put(self, item):
        key = self.bucketing_function(item)
        self.buckets[key].append(item)
        if len(self.buckets[key]) >= self.batch_size:
            self.topqueue.put(self.buckets[key])
            self.buckets[key] = []

    def batch_available(self):
        return not self.topqueue.empty()

    def get(self):
        return self.topqueue.get()

    def queue_all(self):
        # Move all items from the buckets to the topqueue, grouped as best
        # as possible.
        # Requires an orderable result from bucketing_function.
        all_items = sum([self.buckets[i] for i in self.buckets.keys()], [])
        self.buckets = defaultdict(list)

        all_items.sort(key=self.bucketing_function)

        t = 0
        while t+self.batch_size < len(all_items):
            self.topqueue.put(all_items[t: t+self.batch_size])
            t += self.batch_size


def generate_samples_from_dir(rootdir, scale=True):
    # Walk through the strokes data directory.
    # For each file, find the corresponding file in the character data dir
    # and read it.
    # Then add the text to the strokesets, and yield them as samples.
    strokes_data_dir = os.path.join(rootdir, "strokes_data")

    if scale:
        scale = read_strokesets.get_data_scale(strokes_data_dir)

    """TODO: Implement a shuffle of the data each iteration by
    shuffling the result of os.walk here. This will only shuffle on a
    per-subdirectory basis but that should be fine, and easy to implement.

    Note that looping through dictionary keys later on also shuffles the data
    within each subdirectory."""
    for dirname, subdirs, files in os.walk(strokes_data_dir):
        if len(files) > 0:

            # group files by instance, by matching r"x##-###(x)?".
            instance_map = defaultdict(dict)
            for f in files:
                if not (f.startswith('.')) and not f == PARAMS.data_scale_file:
                    instance, num = re.match(r"(\w\w\w-\w\w\w\w?)-(\w\w)", f).group(1,2)
                    num = int(num)
                    instance_map[instance][num] = os.path.join(dirname, f)


            relative_dirname = os.path.relpath(dirname, strokes_data_dir)

            # For each identified instance, read the text from the character_data
            # directory.
            # Then read the strokesets from the files and match them to the
            # text data.
            for charset_name in instance_map.keys():
                char_path = os.path.join(rootdir, "character_data", relative_dirname, charset_name + ".txt")

                textobj = read_chardata.text_from_file(char_path)

                text_dict = textobj.coded_lines
                strokefile_dict = instance_map[charset_name]
                # Now text_dict is a dictionary looking up numbers to text and
                # strokefile_dict is a dictionary looking up
                # numbers to strokeset filenames.

                # All we need to do is loop through the keys in the strokeset
                # dict, read the strokesets and yield the char/strokes pair
                # (combined into a Sample object).

                for linenum in text_dict.keys():
                    textline = text_dict[linenum]
                    strokeset = read_strokesets.strokeset_from_file(
                                                   strokefile_dict[linenum],
                                                   data_scale=scale)
                    #strokeset = None
                    """TODO: IMPLEMENT DATA SCALING"""
                    sample = Sample(strokeset, textline, charset_name, linenum)
                    yield sample

def long_minibatch_from_samples(list_of_samples):
    character_sequences = np.zeros([len(list_of_samples), PARAMS.max_char_len])
    for i in range(len(list_of_samples)):
        character_sequences[i, 0: len(list_of_samples[i].text)] = list_of_samples[i].text

    character_lengths = np.array([len(list_of_samples[i].text)
                                  for i in range(len(list_of_samples))])

    points_generators = [s.strokeset.points_generator(length=10000)
                            for s in list_of_samples]

    sequences = []
    sequence_lengths = []

    for n, sample in enumerate(points_generators):
        next_points = next(sample)
        sequences.append(next_points)
        sequence_lengths.append(len(next_points) - 1)

    offsets_array = np.zeros([len(list_of_samples), max(sequence_lengths) + 1, 3])

    for n, points in enumerate(sequences):
        offsets_array[n, 0: len(points), :] = points

    return Minibatch(offsets_array,    character_sequences,
                    sequence_lengths, character_lengths)

def generate_minibatches_from_samples(list_of_samples):
    # Takes in a list of samples
    # At each step it gets the next PARAMS.sequence_len points from the
    # samples and stacks them over each other,
    # does the same with next points.
    # Feeds them back in a minibatch object, along with the other
    # data such as sequence length and character data.

    # Fill in the character data array. Initialise it as zeros, then pour in
    # the data from the samples.
    character_sequences = np.zeros([len(list_of_samples), PARAMS.max_char_len])
    for i in range(len(list_of_samples)):
        character_sequences[i, 0: len(list_of_samples[i].text)] = list_of_samples[i].text

    character_lengths = np.array([len(list_of_samples[i].text)
                                  for i in range(len(list_of_samples))])

    points_generators = [s.strokeset.points_generator(length=PARAMS.sequence_len)
                            for s in list_of_samples]
    finished = [False for s in list_of_samples]

    # while there are unfinished samples:
    while sum(finished) < len(finished):
        # Initialise zeros offsets array
        # and empty sequence lengths array
        offsets_array = np.zeros([len(list_of_samples), PARAMS.sequence_len + 1, 3])
        sequence_lengths = np.zeros([len(list_of_samples)])

        # For each sample in the batch, if not finished, grab the next points
        # set from the generator.
        for n, sample in enumerate(points_generators):
            if not finished[n]:
                next_points = next(sample)

                # If it's length < sequence_len+1, set the sample to finished.
                if len(next_points) < PARAMS.sequence_len + 1:
                    finished[n] = True

                # Pour the points set into the offsets array
                offsets_array[n, 0: len(next_points), :] = next_points

                # Add the length to the sequence lengths array
                # Subtract one because we count pairs of subsequent
                # points to make inputs/outputs pairs.
                sequence_lengths[n] = len(next_points) - 1

        yield Minibatch(offsets_array,    character_sequences,
                        sequence_lengths, character_lengths)

def generate_minibatches_from_dir(directory, scale=True):
    def make_bucketing_function(bucket_width):
        def bucketing_function(sample):
            return int(round(sample.strokeset.length()/
                             float(bucket_width))
                       * float(bucket_width))
        return bucketing_function

    samples_generator = generate_samples_from_dir(directory, scale=scale)
    minibatch_queue = BatchBucketQueue(
                                make_bucketing_function(PARAMS.bucket_width),
                                PARAMS.batch_size)

    while True:
        try:
            nextsample = next(samples_generator)
            minibatch_queue.put(nextsample)

            if minibatch_queue.batch_available():
                list_of_samples = minibatch_queue.get()
                yield long_minibatch_from_samples(list_of_samples)

        except StopIteration:
            break

    # Push all remaining samples onto the queue and yield them up.
    minibatch_queue.queue_all()
    while minibatch_queue.batch_available():
        list_of_samples = minibatch_queue.get()
        yield long_minibatch_from_samples(list_of_samples)


if __name__ == "__main__":
    # print(max([len(sample.text) for sample in generate_all_from_dir(PARAMS.samples_directory)]))

    """
    firstsample = next(generate_samples_from_dir(PARAMS.samples_directory))
    print(firstsample.strokeset.length())
    print([len(arr) for arr in firstsample.strokeset.points_generator()])"""

    for i in generate_minibatches_from_dir(PARAMS.samples_directory):
        #print(i.offsets_data.shape)#[:,0,:])
        print(i.sequence_lengths)#.shape)
        #print(i.outputs_data.shape)
        #print(i.text_lengths)
        print("------")
