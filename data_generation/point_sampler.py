"""
point_sampler.py
Module for sampling points from the surface of shapes.

Tim Player playertr@oregonstate.edu May 23,2021
"""
from abc import ABC
import numpy as np
from shapely.geometry import LineString

class PointSampler(ABC):
    def sample_points(self, shape):
        """Sample points at the surface of this shape.
        """
        raise NotImplementedError("PointSampler abstract base class method.")

class UniformPointSampler(PointSampler):
    def sample_points(self, shape, n_points):
        """Sample points along surface at uniformly distributed random rays from
        the origin. Note: assumes object surrounds the origin.

        NOTE: shapely's intersections have rounding error.

        Args:
            shape (Polygon): Polygon object surrounding the origin
            n_points (int) : number of points to sample

        Returns:
            np.ndarray: (n_points, 2) array of 2D points at surface
        """
        pts = np.zeros((n_points, 2))
        for i in range(n_points):
            theta = 2*np.pi * np.random.rand()
            radius = 99999
            line = LineString([
                (0, 0), 
                (radius*np.cos(theta), radius*np.sin(theta))])
            x, y = list(shape.intersection(line).coords)[1]
            pts[i,:] = x, y
        return pts