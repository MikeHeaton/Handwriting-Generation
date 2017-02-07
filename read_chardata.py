import os
from read_strokesets import load_xml_from_file
from config import PARAMS

class Text:
    def __init__(self, textdict):
        self.lines = textdict

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
    CSR_location = content.index("CSR:")
    textlines = content[CSR_location + 2:]

    textdict = {i+1: c for i, c in enumerate(textlines)}

    """xml_data = load_xml_from_file(filepath)
    textdict = load_textdict_from_xml(xml_data)"""

    return Text(textdict)

if __name__ == "__main__":
    charpath = os.path.join(PARAMS.samples_directory, "character_data", "a01", "a01-001", "a01-001w.txt")
    strokepath = os.path.join(PARAMS.samples_directory, "strokes_data", "a01", "a01-001", "a01-001w-01.xml")
    filepath = os.path.join(PARAMS.samples_directory, "character_data", "a01", "a01-001", "a01-001w.txt")
    print(text_from_file(charpath).text)
