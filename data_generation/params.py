"""
params.py
Global parameters for data generation code.

Tim Player playertr@oregonstate.edu June 1, 2021
"""
from math import pi

class TrainingExample:
    N_SCAN_POINTS = int(100)
    SCAN_HALF_ANGLE = 30 * pi/180 # radians
    SCAN_RANGE = 99999
    FOV_REDUCTION_MARGIN = 0.0001 # radians
    FRICTION_COEFFICIENT = 0.3

class DataGen:
    N_POLYGONS = 200#int(1e3)        # number of different shapes
    N_PERSPECTIVES = 50    # number of different robot perspectives
    MIN_DISTANCE = 1        # minimum distance from robot loc to shape
    ROBOT_RANGE = [-10, 10, -10, 10]   # bounds min_x, max_x, min_y, max_y for robot
    VIEW_RANGE = [MIN_DISTANCE, 15, -5, 5]
    N_SDF_QUERIES = 1000
    GRASP_BOUNDS = [0, 2*pi, -6, 6] # theta, b boundaries for uniform gen
    NUM_GRASPS = 500
    PROPORTION_CONTACT_GRASPS = 0.7 # percentage of training grasps that contact object
    BAD_GRASP_QUALITY = -5 # qualtiy of noncontact grasps
