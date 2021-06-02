"""
utils.py
Module for utility/helper functions used in data generation.

Tim Player playertr@oregonstate.edu May 23,2021
"""

def plot_shape(shape, ax, title="Shape", bounds=[-4, 4, -4, 4], **plt_kwargs):
    """ Plot the shape on an axis."""
    
    ax.set_title(title)
    ax.plot(*shape.exterior.xy, **plt_kwargs)
    ax.set_xlim(bounds[0:2])
    ax.set_ylim(bounds[2:])
    ax.set_aspect('equal')