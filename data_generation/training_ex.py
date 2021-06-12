"""
training_ex.py
Module for simulating the LIDAR of a 2D robot from different perspectives, as well as collecting ground truth SDF functions.

Tim Player playertr@oregonstate.edu June 1, 2021
"""

import numpy as np
from . import params
from shapely.geometry import LineString, MultiLineString, Point
from .grasp_quality import shape_grasp_quality
from shapely.geometry import Polygon

class TrainingExample:
    def __init__(self, robx, roby, shape):
        self.robx       = robx
        self.roby       = roby
        self.shape      = shape
        self.robang     = None # initialized in set_scan_data()
        self.scan_pts   = None # initialized in set_scan_data()

        self.set_scan_data()

        coords = np.array(self.shape.exterior.coords)
        self.tf_shape = Polygon(rotate(coords, -self.robang))

    def set_scan_data(self):
        """ Get a local-frame laser scan, its angle, and a signed distance function. """

        pts, self.robang = self.scan()
        self.scan_pts = glob_to_loc(pts, self.robang, self.robx, self.roby)

    def scan(self,
    n_points=params.TrainingExample.N_SCAN_POINTS,
    s_range=params.TrainingExample.SCAN_RANGE,
    eps=params.TrainingExample.FOV_REDUCTION_MARGIN):
        """Cast N_SCAN_POINTS evenly-spaced rays, but they ALL land on the shape.
        """
        # Identify shape boundary
        cw_theta, diff = vis_bounds(self.robx, self.roby, self.shape)
        thetas = np.linspace(cw_theta+eps, cw_theta+diff-eps, n_points)
        
        pts = np.zeros((n_points, 2))

        for i, t in enumerate(thetas):
            line = LineString([
                (self.robx, self.roby), 
                (self.robx+s_range*np.cos(t), self.roby+s_range*np.sin(t))])
            try:
                intersection = self.shape.intersection(line)
                if type(intersection) == MultiLineString:
                    x, y = list(self.shape.intersection(line)[0].coords)[0]

                else:
                    x, y = list(self.shape.intersection(line).coords)[0]
            except IndexError:
                print(f"No intercept: theta={t}")
                x, y = np.nan, np.nan

            pts[i,:] = x, y
        
        return pts, wrap(cw_theta + diff/2)

    def sdf(self, x):
        return local_sdf(self.shape, x, self.robang, self.robx, self.roby)

    def grasp_quality(self, theta, b):
        return shape_grasp_quality(self.tf_shape, theta, b, mu=params.TrainingExample.FRICTION_COEFFICIENT)

def vis_bounds(robx, roby, shape):
    """Find angle range at which the shape is visible to a location.

    Returns:
        cw, diff (float, float): rightmost (CW) angle and angular extent through which shape is visible
    """

    # TODO: beat O(N^2), see https://stackoverflow.com/questions/19074717/finding-the-largest-difference-between-angles-in-an-array

    # find angle from point to each vertex
    verts = [(x,y) for x,y in zip(*shape.exterior.xy)]
    thetas = np.array(sorted(
        [np.arctan2(y_-roby,x_-robx) for x_,y_ in verts]
    ))

    # find maximum pairwise angle difference
    def angle_diff(t1, t2):
        """Smallest angle-difference between two angles"""
        diff = np.abs(t1 - t2)
        return min(diff, 2*np.pi - diff)

    # identify (sorted) pair of angles with largest difference
    best = (-1, -1, -1)
    for i in range(len(thetas)):
        for j in range(i+1, len(thetas)):
            diff = angle_diff(thetas[i], thetas[j])
            if diff > best[2]:
                best = (thetas[i], thetas[j], diff)
    
    # return furthest-clockwise angle first, such that
    # (cw + dtheta) % np.pi == ccw, with dtheta <= np.pi
    if angle_equal(best[0]+best[2], best[1]):
        cw = best[0]
    else:
        cw = best[1]
    
    return cw, best[2]

def global_sdf(shape, ptsg):
    points = [Point(pt[0], pt[1]) for pt in ptsg]
    def signed_distance(shape, point):
        dist = shape.exterior.distance(point)
        if point.within(shape):
            dist = dist*-1
        return dist
    signed_distances = [signed_distance(shape, pt) for pt in points]
    return np.array(signed_distances)

def local_sdf(shape, pts, robang, robx, roby):
    ptsg = loc_to_glob(pts, robang, robx, roby)
    return global_sdf(shape, ptsg)

def glob_to_loc(pts, robang, robx, roby):
    return rotate(translate(pts, -robx, -roby), -robang)

def loc_to_glob(pts, robang, robx, roby):
    return translate(rotate(pts, robang), robx, roby)

def translate(pts, dx, dy):
    """Translate an Mx2 ndarray of points by dx, dy

    Args:
        pts (ndarray): Mx2
        dx (float): distance to move in +x dir
        dy (float): distance to move in +y dir

    Returns:
        ndarray: Mx2 ndarray after translation
    """
    return pts + np.array([[dx, dy]])

def rotate(pts, theta):
    """Rotate an Mx2 ndarray of points through angle theta about origin.

    Args:
        pts (ndarray): Mx2
        theta (float): radians CCW

    Returns:
        ndarray: Mx2 ndarray after rotation
    """
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return (rot_mat @ pts.T).T

def wrap(angle):
    if -np.pi < angle < np.pi:
        return angle
    else:
        return ((angle + np.pi) % (2*np.pi)) - np.pi

def angle_equal(a1, a2, eps=0.001):
    # determines whether two angles are equal, but also gives 7*np.pi/6 = -5*np.pi/6
    return np.abs(((a1 + np.pi) % (2*np.pi)) - ((a2 + np.pi) % (2*np.pi))) < eps

############# DEPRECATED ######################################################
    # def wide_scan(self, shape, 
    #     n_points=params.TrainingExample.N_SCAN_POINTS,
    #     half_angle=params.TrainingExample.SCAN_HALF_ANGLE,
    #     s_range=params.TrainingExample.SCAN_RANGE):
    #     """Cast N_SCAN_POINTS evenly-spaced rays out from the robot at a fixed fan angle. Rays that miss are given (nan, nan).
    #     """

    #     pts = np.zeros((n_points, 2))

    #     thetas = np.linspace(
    #         self.theta-half_angle, 
    #         self.theta+half_angle, 
    #         n_points)

    #     for i, t in enumerate(thetas):
    #         line = LineString([
    #             (self.x, self.y), 
    #             (self.x+s_range*np.cos(t), self.y+s_range*np.sin(t))])
    #         try:
    #             x, y = list(shape.intersection(line).coords)[0]
    #         except IndexError:
    #             x, y = np.nan, np.nan
    #         pts[i,:] = x, y
    #     return pts