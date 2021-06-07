"""
generate_data.py
Script for generating training data.

Training data consists of single-view LIDAR scans producing local-frame cartesian coordinates, obtained by casting N_SCAN_POINTS rays that evenly land on the shape surface. The shapes are all random pentagons.

For each shape, N_PERSPECTIVES different robot positions are chosen.

Ground truth is provided as a matrix of vertices.

Tim Player playertr@oregonstate.edu May 23,2021
"""

from data_generation.shape_generator import PolygonGen
from data_generation.training_ex import TrainingExample
import data_generation.params as params
from shapely.geometry import Point
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import itertools
import argparse
import pickle

## Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='data',
                    help="Directory into which to write the dataset")

print("Generating Data. See data_generation/params.py to modify data properties.")
args = parser.parse_args()
output_dir = args.output_dir

## Generate random polygons
print("Generating random polygons.")
pg = PolygonGen()
shapes = pg.get_shapes(5, params.DataGen.N_POLYGONS)

## Collect rendered views of each polygon as well as SDF fields
def collect_examples(shape):
    examples = []
    for _ in range(params.DataGen.N_PERSPECTIVES):
        min_x, max_x, min_y, max_y = params.DataGen.ROBOT_RANGE

        # Sample workspace uniformly for robot location, but filter by distance
        # from shape
        while True:
            x = np.random.rand() * (max_x-min_x) - (max_x-min_x)/2
            y = np.random.rand() * (max_y-min_y) - (max_y-min_y)/2

            if shape.exterior.distance(Point(x,y)) > params.DataGen.MIN_DISTANCE:
                break
        
        te = TrainingExample(robx=x, roby=y, shape=shape)
        examples.append(te)
    return examples

print("Rendering training examples from polygons.")
# Collect training examples from shapes in parallel
with Pool(cpu_count()-1) as p:
    examples = list(
        tqdm(
            p.imap(collect_examples, shapes),
            total=len(shapes)
        )
    )

examples = list(itertools.chain(*examples))

# Each example is a TrainingExample object. A TrainingExample encodes a scene with a randomly positioned robot, and it has the following state and behavior:
# robx, roby    : the x and y position of the robot
# shape         : the Polygon object (in the global frame), centered at (0,0)
# robang        : the angle from the robot to the center of the visible region of the shape
# scan_pts      : an (N_SCAN_POINTS,2) ndarray of coordinates corresponding to the local-frame cartesian locations obtained by casting rays from the robot to the shape. The rays span the entire visible portion of the shape.
# sdf(x)        : a vectorized function converting an ndarray of coordinates (x,y) expressed in the robot local frame, to distance from the object. This function is used to evaluate the ground truth SDF.
# grasp_quality(theta, b) : a function giving the quality of a grasp with angle theta and offset b, on the object's `shape` attribute.

## Write training data
print("Pickling training examples.")
pickle.dump(examples, open(
    output_dir+"/examples.p", "wb"))

print("Complete.")