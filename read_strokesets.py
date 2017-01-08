import xml.etree.ElementTree as ET
import numpy as np
import os
import matplotlib.pyplot as plt
from config import PARAMS

class StrokeSet(object):
    def __init__(self, strokes, fname=""):
        self.strokes = strokes
        self.fname = fname

    def __str__(self):
        return str([str(s) for s in self.strokes])

    def to_numpy(self):
        # Numpy array dimension Nx3
        # columns x,y,q
        # q is 1 if point is last in a stroke, else 0.

        # Collect all points OFFSET data from strokes into a numpy array

        self.all_points = []
        self._last_point = Point(0.0, 0.0)

        def append_offset(point, is_end_stroke):
            self.all_points.append((point.x - self._last_point.x,
                                    point.y - self._last_point.y,
                                    is_end_stroke))
            self._last_point = point

        for stroke in self.strokes:
            for p in stroke.points[:-1]:
                append_offset(p, 0)
            append_offset(stroke.points[-1],1)

        self.all_points = np.array(self.all_points[1:])
        return self.all_points

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', 'datalim')
        ax.invert_yaxis()
        jet = plt.get_cmap('jet')
        for x, stroke in enumerate(self.strokes):
            coordinates = [p.coordinates() for p in stroke.points]
            x_points, y_points = zip(*coordinates)
            ax.scatter(x_points, y_points, s=1, c=jet(100*x % 256), lw=0)
        fig.savefig('testplot.png')
        """TODO: clean this up. Put 'plot' methods in the strokes?"""

        return fig, ax

class Stroke(object):
    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str([str(p) for p in self.points])

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def coordinates(self):
        return (self.x, self.y)

def load_xml_from_file(filename):
    tree = ET.parse(filename)
    return tree

def load_strokeset_from_xml(xml_data):
    root = xml_data.getroot()
    strokeset_node = root[1]

    all_strokes = []
    for stroke_node in strokeset_node:
        points = [Point(float(point.attrib['x'])*PARAMS.data_scale, float(point.attrib['y'])*PARAMS.data_scale)
                    for point in stroke_node]
        stroke = Stroke(points)
        all_strokes.append(stroke)
    stroke_set = StrokeSet(all_strokes)
    """TODO: make this neater with a comprehension"""

    return stroke_set

def all_strokesets_from_dir(rootdir, max_strokesets=None):
    all_strokesets = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            xml_data = load_xml_from_file("{}/{}".format(subdir,file))
            stroke_set = load_strokeset_from_xml(xml_data)
            stroke_set.fname = file
            all_strokesets.append(stroke_set)
            if max_strokesets is not None and len(all_strokesets) >= max_strokesets:
                break
        if max_strokesets is not None and len(all_strokesets) >= max_strokesets:
            break

    return all_strokesets
