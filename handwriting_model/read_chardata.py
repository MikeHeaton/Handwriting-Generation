import os
import read_strokesets
from config import PARAMS

class Text:
    def __init__(self, textdict):
        self.lines = textdict
        self.coded_lines = {line: [PARAMS.char_to_int[c] for c in textdict[line]]
                            for line in textdict.keys()}

    """TODO: Clean the text re " . " (space . space) and etc."""

def load_textdict_from_xml(xml_data, data_scale=False):
    root = xml_data.getroot()
    characters_node = root[1]

    textlines = [c.attrib['text'] for c in characters_node if c.tag == "TextLine"]
    textdict = {i+1: c for i, c in enumerate(textlines)}

    return textdict

def text_from_file(filepath):
    # Reads the character data from a file and returns it in a Text object.
    with open(filepath) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    # Find the "CSR:" header, and read everything from two lines below it,
    # through to the text end (one line before file end).
    # Filter out blank lines.
    CSR_location = content.index("CSR:")
    textlines = [c for c in content[CSR_location + 2:] if len(c) > 0]

    textdict = {i+1: c for i, c in enumerate(textlines)}

    """xml_data = load_xml_from_file(filepath)
    textdict = load_textdict_from_xml(xml_data)"""

    return Text(textdict)

if __name__ == "__main__":
    charpath = os.path.join(PARAMS.samples_directory, "character_data", "a02", "a02-057", "a02-057.txt")
    strokepath = os.path.join(PARAMS.samples_directory, "strokes_data", "a02", "a02-057", "a02-057-08.xml")
    print(text_from_file(charpath).lines)
    read_strokesets.strokeset_from_file(strokepath).plot()
