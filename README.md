# Surface Reconstruction and Grasp Synthesis

## Overview

This project explores 2D object reconstruction and planar grasp synthesis using the deep learning architecture from PointSDF.

## Installation
Use conda or pip to install the following requirements in an environment, if you don't already have them. I didn't have a requirements.txt file because some of them are tough to install. For [Pytorch](https://pytorch.org/), you'll probably want to set up CUDA, which is slightly involved.
```
numpy
tqdm
scipy
shapely
```

## Usage
If you need to generate new data, use, e.g., 
```
$ python generate_data.py --output_dir my_dir
```

For other parameters, modify data_generation/params.py.

Training and testing with the neural network TBD.

## Understanding the generated data
The data in `examples.p` consists of many TrainingExample objects, which are defined in `data_generation/training_ex.py`. 

A TrainingExample encodes a scene with a randomly positioned robot, and it has the following state and behavior:
* robx, roby    : the x and y position of the robot
* shape         : the Polygon object (in the global frame), centered at (0,0)
* robang        : the angle from the robot to the center of the visible region of the shape
* scan_pts      : an (N_SCAN_POINTS,2) ndarray of coordinates corresponding to the local-frame cartesian locations obtained by casting rays from the robot to the shape. The rays span the entire visible portion of the shape.
* sdf(x)        : a vectorized function converting an ndarray of coordinates (x,y) expressed in the robot local frame, to distance from the object. This function is used to evaluate the ground truth SDF.
* grasp_quality(theta, b) : a function giving the quality of a grasp with angle theta and offset b, on the object's `shape` attribute.

It is unconventional to store training data as Python objects with functions, but in our case it makes sense because the network must learn two functions (SDF and grasp quality).

***I highly recommend looking at the source code to understand the TrainingExample's state and behavior.*** In particular:
* The data generation script in `generate_data.py`
* The data generation parameters in `data_generation/params.py`
* The example usage in `notebooks/scratchwork/test_data.ipynb`
* The class definition in `data_generation/training_ex.py`

may all prove useful, in that order.

## Structure
The code is divided into separate  `data_generation` and `network` folders as follows.

```
PlanarGrasping/
│   README.md
│   generate_data.py                # Script to generate training data
│
└───data_generation/                # Package supporting data generation
│   │   __init__.py
│   │   grasp_quality.py            # Assessing grasps on 2D polygons
│   │   params.py                   # Parameters for data generation
│   │   shape_generator.py          # Random polygon generation
│   │   training_ex.py              # TrainingExample class holding data state
│   │   utils.py                    # Plotting utilities
|
└───notebooks/                      # IPython notebooks for development
│   └───scratchwork                 
|   |   |   ...   

##################### DOES NOT EXIST YET; LEFTOVER FROM GPNN ##############
└───network
|   │   evaluate.py                 # Module + script for testing/validation
|   │   search_hyperparams.py       # Script for tuning params
|   │   synthesize_results.py       # Script for summarizing saved metrics
|   │   train.py                    # Module + script for training network
|   │   utils.py                        
|   |
|   └───data                        # Pickle files with partitioned data
|   |   └───test_sdf 
|   |   | 
|   |   └───train_sdf
|   |   | 
|   |   └───val_sdf
|   |   
|   └───experiments                 # Folders for trying different parameters
|   |   └───sdf_cnn
|   |
|   └───model
|   |   |   data_loader.py          # Pytorch dataloader class for train/test
|   |   |   net.py                  # Pytorch neural network
```