"""
generate_data.py
Script for generating training data.

Tim Player playertr@oregonstate.edu May 23,2021
"""

from data_generation.shape_generator import PolygonGen
from data_generation.point_sampler import UniformPointSampler
from data_generation.gp import VanillaGP
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

## Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='network/data',
                    help="Directory into which to write the dataset")
parser.add_argument('--num_examples', type=int, default=100000,
                    help="How many examples to generate")

args = parser.parse_args()
output_dir = args.output_dir

## Define constants
N_POLYGONS = args.num_examples   # Number of polygons/training examples to generate data for
N_GP_TRAINING_PTS = 500 # Number of points sampled from each polygon to train GP
N_SURFACE_RAYS = 100    # Number of evenly spaced rays used to represent zero level set

## Generate random polygons
pg = PolygonGen()
shapes = pg.get_shapes(5, N_POLYGONS)

## Retrieve zero-level surface points from GPs trained on these random polygons

# First, we define a function that samples points from a polygon's surface,
# trains a GP to approximate the polygon, and then identifies the "zero
# level set".

def training_data_and_surface(shape):
    """Train a GP on this shape and return the sampled input points, scalar
    observations `y` for each input point, and the zero level set denoting the
    object's surface.

    The zero level set is defined as the distance from the origin to the SDF
    zero along many evenly-spaced rays from the origin.

    Args: shape (Polygon): shape to train GP on

    Returns: (tuple of ndarrays): pts is a 
    """

    # Sample points from the shape surface to train the GP
    ps = UniformPointSampler()
    pts = ps.sample_points(shape, N_GP_TRAINING_PTS)
    y = np.zeros((pts.shape[0]))

    # Set a point at the origin to have negative SDF
    pts[0] = [0, 0]
    y[0] = -1

    # Train the GP and the distance to the regressed surface along
    # evenly-spaced rays from the origin.
    gp = VanillaGP(noise=0.02, l=0.5, sig_var=1)
    gp.train(pts, y)
    surface_pts = gp.get_surface(N_SURFACE_RAYS)

    return (pts, y, surface_pts)

# It is expensive to retrieve the training data and surface points from all
# of the shapes. To speed this up, we will use a multiprocessing.Pool to
# compute the training data and surface points from shapes in each chunk in
# parallel.

# tqdm gives us a progress bar.

with Pool(cpu_count()-1) as p:
    examples = list(
        tqdm(
            p.imap(training_data_and_surface, shapes),
            total=len(shapes)
        )
    )

# `examples` is a list of tuples. Each tuple has three ndarrays: `pts`, `y`,
# and `surface_pts`.

print("Finished retrieving ground truth zero level sets and training data.")


## Shuffle data and partition into train, test, and validation sets
import random
random.seed(112)
random.shuffle(examples)

split1 = int(0.8 * len(examples))
split2 = int(0.9 * len(examples))

train_examples = examples[:split1]
val_examples   = examples[split1:split2]
test_examples  = examples[split2:]

## Write data into its home in the network/ folder
pickle.dump(train_examples, open(
    output_dir+"/train_sdf/sdf_train_examples.p", "wb"))
pickle.dump(val_examples, open(output_dir+"/val_sdf/sdf_val_examples.p", "wb"))
pickle.dump(test_examples, open(output_dir+"/test_sdf/sdf_test_examples.p", "wb"))