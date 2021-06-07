"""
utils.py
Module for utility/helper functions used in data generation.

Tim Player playertr@oregonstate.edu May 23,2021
"""
import numpy as np

def plot_shape(shape, ax, title="Shape", bounds=[-10, 10, -10, 10], **plt_kwargs):
    """ Plot the shape on an axis."""
    
    ax.set_title(title)
    ax.plot(*shape.exterior.xy, **plt_kwargs)
    ax.set_xlim(bounds[:2])
    ax.set_ylim(bounds[2:])
    ax.set_aspect('equal')

def plot_sdf(sdf, ax, title='SDF', bounds=[0,10*np.sqrt(2),-5,5], **plt_kwargs):
    """ Plot the signed distance function """
    x = np.linspace(*bounds[:2], 100)   # (100,) vector
    y = np.linspace(*bounds[2:], 100)   # (100,) vector
    X, Y = np.meshgrid(x, y)            # Two (100,100) matrices
    pts = np.column_stack([X.ravel(), Y.ravel()])   # (10000,2) matrix

    dists = sdf(pts).reshape((len(x),len(y))) # (100,100) matrix of distances
    ax.contourf(x, y, dists, 20, cmap='cividis', **plt_kwargs)
    ax.set_title(title)

def plot_training_example(te, axs):
    plot_sdf(te.sdf, axs[0], title="Rob-Frame Scan and SDF")
    axs[0].scatter(te.scan_pts[:,0], te.scan_pts[:,1])
    axs[0].set_aspect('equal')

    plot_shape(te.shape, axs[1], title="Glob-Frame Robot and Shape")
    axs[1].scatter(te.robx, te.roby, color='orange')
    axs[1].quiver(te.robx, te.roby, np.cos(te.robang), np.sin(te.robang), scale=1)
    axs[1].set_aspect('equal')