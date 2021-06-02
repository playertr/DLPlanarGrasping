"""
gp.py
Module for training and inference on Gaussian Processes.

Tim Player playertr@oregonstate.edu May 23,2021
"""
from abc import ABC
import numpy as np
import sklearn.metrics
import scipy

class GP(ABC):
    """Abstract Base Class for Gaussian Process.
    """

    def train(self, x, y):
        raise NotImplementedError("GP abstract base class method.")
        return K, x, y, fn

    def kernel(self, a, b, *args):
        raise NotImplementedError("GP abstract base class method.")

class VanillaGP(GP):
    """Standard Gaussian Process, not sparse.
    """

    def __init__(self, offset=5, noise=0, l=1, sig_var=1):
        self.offset = offset
        self.noise = noise
        self.l=l
        self.kernel = self.get_rbf_kernel(l=l, sig_var=sig_var)


    def train(self, Xtrain, ytrain):
        """Trains a Gaussian Process on this data matrix Xtrain and observation vector ytrain.

        Args:
            Xtrain (np.ndarray): Training input data. Each row is an observation input x.
            ytrain (np.ndarray): Training observation vector. Each element is the scalar observation associated with a row from Xtrain.

        Returns:
            K, Xtrain, ytrain, test_fn, mu_fn: The first three returns are the Gram matrix and the input training data. `test_fn` is a function that will evaluate the GP mean and covariance at a new test point. `mu_fn` is a function that will evaluate the GP mean at a new test point.
        """
        # Apply offset to bias the mean. See Rasmussen and Williams p. 27
        ytrain = ytrain - self.offset
        self.Xtrain = Xtrain
        # Apply kernel function to training points
        K = self.kernel(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + self.noise*np.eye(K.shape[0]))
        self.K_inv_y = np.linalg.solve(L.T, np.linalg.solve(L, ytrain))

        def test_fn(Xtest):
            K_ss = self.kernel(Xtest, Xtest)
            K_s = self.kernel(Xtrain, Xtest)
            # compute mean
            mu = (K_s.T@self.K_inv_y).squeeze() + self.offset # incorporate biased mean
            v = np.linalg.solve(L, K_s)
            # compute covariance matrix
            cov = K_ss-v.T@v
            return mu, cov

        def mu_fn(Xtest):
            K_s = self.kernel(Xtrain, Xtest)
            # compute mean
            mu = (K_s.T@self.K_inv_y).squeeze() + self.offset # incorporate biased mean
            return mu

        self.test_fn = test_fn
        self.mu_fn = mu_fn
        return K, Xtrain, ytrain, test_fn, mu_fn

    def get_surface(self, N):
        """Retrieves a vector of distances from the origin to the surface (zero-level set) along N evenly spaced rays.

        Args:
            N (int): number of evenly spaced rays
            method (str): solution method for numpy rootfinder
        """

        def raytrace_zero(theta):
            """ Returns the first root of the SDF function
            encountered along the ray from the origin at angle theta. """
            # Get SDF function along the ray
            c, s = np.cos(theta), np.sin(theta)
            def sdf_along_ray(r):
                """ Evaluate SDF along ray from origin at polar coords r, theta.
                """
                return self.mu_fn(np.array([[r*c, r*s]]))

            # Return the zero crossing of the constrained SDF.
            sol = scipy.optimize.root_scalar(sdf_along_ray, x0=0.5, bracket=[0,5])
            if not sol.converged:
                raise Exception("Could not find zero-crossing of SDF.")
            return sol.root
            
        thetas = np.array(range(N)) * 2*np.pi/N
        return np.array([raytrace_zero(t) for t in thetas])

    def get_rbf_kernel(self, l, sig_var):
        """Returns the RBF kernel function with parameters l, sig_var.
        """
        def kernel(a, b):
            distance = sklearn.metrics.pairwise_distances(a,b)
            return sig_var*np.exp(-1/(2*l**2)*distance**2)
        return kernel

    def plot_gp(self, ax, fig, bounds=[-4, 4, -4, 4], title="GP"):
        """Plot the value of the GP over a 2D grid, with a zero-level contour.
        """
        x_test = np.linspace(bounds[0], bounds[1])
        y_test = np.linspace(bounds[2], bounds[3])
        g = np.meshgrid(x_test, y_test)
        Xtest = np.vstack(list(map(np.ravel, g))).T

        mu, cov = self.test_fn(Xtest)
        contourf = ax.contourf(x_test, y_test, mu.reshape(len(x_test), len(y_test)), 20, cmap='cividis')
        ax.contour(x_test, y_test, mu.reshape(len(x_test), len(y_test)), levels=[0], colors='red')
        ax.set_title(title)
        fig.colorbar(contourf, ax=ax)