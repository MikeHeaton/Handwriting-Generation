import read_chardata
import read_strokesets
import os
from config import PARAMS
import re
from collections import defaultdict

def generate_all_from_dir(rootdir):
    # Walk through the strokes data directory.
    # For each file, find the corresponding file in the character data dir
    # and read it.
    # Then add the text to the strokesets, and yield them as samples.
    strokes_data_dir = os.path.join(rootdir, "strokes_data")
    for dirname, subdirs, files in os.walk(strokes_data_dir):
        if len(files) > 0:

            # split files up by which instance matching r"x##-###(x)?" they're part of.
            instance_map = defaultdict(dict)
            for f in files:
                if not (f.startswith('.')) and not f == PARAMS.data_scale_file:
                    instance, num = re.match(r"(\w\w\w-\w\w\w\w?)-(\w\w)", f).group(1,2)
                    num = int(num)
                    instance_map[instance][num] = os.path.join(dirname, f)

            relative_dirname = os.path.relpath(dirname, strokes_data_dir)

            for charset_name in instance_map.keys():
                char_path = os.path.join(rootdir, "character_data", relative_dirname, charset_name + ".txt")

                textobj = read_chardata.text_from_file(char_path)
                print("-------")
                print(textobj.text)
                print(instance_map[charset_name])

                text_dict = textobj.lines
                strokefile_dict = instance_map[charset_name]
                # Now text_dict is a dictionary looking up numbers to text;
                # strokefile_dict is a dictionary looking up
                # numbers to strokeset filenames.

                # All we need to do is loop through the keys in the strokeset
                # dict, read the strokesets and yield the char/strokes pair.

                for linenum in text_dict.keys():
                    textline = text_dict[linenum]
                    strokeset = read_strokesets.strokeset_from_file(
                                                    strokefile_dict[linenum])
                    """TODO: IMPLEMENT DATA SCALING"""

                    


            # now instance_map contains all the characters in keys,
            # with all the strokeset paths as items.




    """
    print(dirname)
    print(os.path.relpath(dirname))
    for filename in files:

        fileid = os.path.split(dirname)[1]
        if not (filename.startswith('.')):
            textobj = read_chardata.text_from_file(os.path.join(dirname, filename), fileid)
            yield textobj"""

def generate_strokesets_from_text(textobject):
    # Constructs the filepath from the text object's fileid.
    # Locates the corresponding stroke files;
    # Reads the strokesets from the files;
    # Attaches the text lines to them;
    # and yields the strokesets.

    # Construct the filepath from the text object's fileid.
    fileid = textobject.id
    regexmatch = re.match(r"(\w+)-(\w+)", fileid)
    id0, id1 = regexmatch.group(0), regexmatch.group(1)


if __name__ == "__main__":
    for text in generate_all_from_dir(PARAMS.samples_directory):
        pass#print(text.text)
