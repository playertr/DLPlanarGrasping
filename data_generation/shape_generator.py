"""
shape_generator.py
Module for creating random shapes.

Tim Player playertr@oregonstate.edu May 23,2021
"""
from abc import ABC
import numpy as np
from shapely.geometry import Polygon

class ShapeGenerator(ABC):
    def get_shapes(self, n_shapes, *args):
        """Creates shapes as a list of Python objects, e.g. shapely.Polygon.

        Args:
            n_shapes (int): number shapes

        Returns:
            list: list of Python objects
        """
        raise NotImplementedError("ShapeGenerator abstract base class method.")

class PolygonGen(ShapeGenerator):
    def get_shapes(self, n_vertices, n_shapes=1):
        """Creates a list of polygons whose vertex thetas are uniformly distributed around [0, 2*pi] and whose vertex r's are normalized to zero.

        Args:
            n_vertices (int): Vertices per polygon
            n_shapes (int, optional): Number of shapes. Defaults to 1.

        Returns:
            list of Polygons: shapely.Polygon objects
        """
        # Should this be a generator function?
        polygons = []
        for _ in range(n_shapes):
            # sample random 2D coords from [0,1]
            rands = np.random.rand(n_vertices, 2)
            # convert theta to range [0, 2*pi] and normalize radius
            polar_verts = np.array([
                2*np.pi,
                1/rands[:,1].mean(axis=0)]
                ) * rands
            # convert to cartesian
            cartesian_verts = np.column_stack([
                polar_verts[:,1] * np.cos(polar_verts[:,0]),
                polar_verts[:,1] * np.sin(polar_verts[:,0])])
            # center
            cartesian_verts = cartesian_verts - cartesian_verts.mean(axis=0)
            # sort by new thetas so shape is non-self-intersecting
            thetas = np.arctan2(cartesian_verts[:,0], cartesian_verts[:,1])
            idxs = np.argsort(thetas)
            cartesian_verts = cartesian_verts[idxs, :]
            polygons.append(Polygon(cartesian_verts))
        return polygons