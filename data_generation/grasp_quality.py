import numpy as np
from scipy.spatial import ConvexHull
from pygel3d import hmesh
import cvxpy as cp
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from data_generation.utils import plot_shape
import matplotlib.pyplot as plt

def cross_matrix(x):
    """
    Returns a matrix x_cross such that x_cross.dot(y) represents the cross
    product between x and y.
    Defined only on 2D vectors, x_cross is a 2x1 vector representing the magnitude of the cross product in the z direction.
    """
    D = x.shape[0]
    if D == 2:
        return np.array([[-x[1], x[0]]])
    raise RuntimeError("cross_matrix(): x must be 2D. Received a {}D vector.".format(D))

def wrench(f, p):
    """
    Computes the wrench from the given force f applied at the given point p.

    Args:
        f - 2D contact force.
        p - 2D contact point.

    Return:
        w - 3D contact wrench represented as (force, torque).
    """
    w = np.concatenate([
        f,
        cross_matrix(p) @ f
    ])
    return w

def cone_edges(f, mu):
    """
    Returns the edges of the specified friction cone. For 3D vectors, the
    friction cone is approximated by a pyramid whose vertices are circumscribed
    by the friction cone.

    In the case where the friction coefficient is 0, a list containing only the
    original contact force is returned.

    Args:
        f - 2D contact force.
        mu - friction coefficient.

    Return:
        edges - a list of forces whose convex hull approximates the friction cone.
    """
    # Edge case for frictionless contact
    if mu == 0.:
        return [f]

    # Planar wrenches
    D = f.shape[0]
    if D == 2:
        edges = []

        # identify line normal to f
        unit_normal = np.array([-f[1], f[0]])
        unit_normal = unit_normal / np.linalg.norm(unit_normal)

        edges.append(f + mu*np.linalg.norm(f)*unit_normal)
        edges.append(f - mu*np.linalg.norm(f)*unit_normal)
        
        return edges
    raise RuntimeError("cone_edges(): f must be 2D. Received a {}D vector.".format(D))

def ferrari_canny(F):
    """Returns the Ferrari Canny metric of a set of wrenches F.
    
    The Ferrari Canny metric is the signed magnitude of the smallest wrench that extends from the origin to the convex hull of the wrenches in the matrix F.

    Args:
        F (ndarray): 3xM matrix of wrenches

    Returns:
        float: distance from origin to hull.
    """
    hull = ConvexHull(F.T)
    origin = np.array([0,0,0])
    return dist(hull, origin)

def dist(hull, point):
    """Return distance from a point to a convex hull.

    Args:
        hull ([type]): [description]
        point ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Construct PyGEL Manifold from the convex hull
    m = hmesh.Manifold()
    for s in hull.simplices:
        m.add_face(hull.points[s])

    dist = hmesh.MeshDistance(m)

    # Get the distance to the point But don't trust its sign, because of
    # possible wrong orientation of mesh face
    d = dist.signed_distance(point)

    # Correct the sign with ray inside test
    if dist.ray_inside_test(point):
        if d > 0:
            d *= -1
    else:
        if d < 0:
            d *= -1

    return float(d)

def form_closure_program(F):
    """
    Solves a linear program to determine whether the given contact wrenches
    are in form closure.

    Args:
        F - matrix whose columns are 3D contact wrenches.

    Return:
        True/False - whether the form closure condition is satisfied.
    """
    if np.linalg.matrix_rank(F) < F.shape[0]:   # deficient -> not form closure
        return False

    k = cp.Variable(F.shape[1])
    objective = cp.Minimize(cp.sum(k))
    constraints = [F@k == 0, k >= 1] # (constraints are elementwise)

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    return prob.status not in ['infeasible', 'unbounded']

def is_in_force_closure(F):
    """
    Calls form_closure_program() to determine whether the wrenches are in force closure.

    Args:
        F - matrix whose columns are 3D contact wrenches from friction cone pyramidal approximation.

    Return:
        True/False - whether the forces are in force closure.
    """
    return form_closure_program(F)

def wrench_basis(forces, points, friction_coeffs):
    """ Construct F, a 3xM matrix of friction cone wrenches. """
    F = []
    for f, p, mu in zip(forces, points, friction_coeffs):
        edge_forces = cone_edges(f, mu)
        wrenches = [wrench(edge_force, p) for edge_force in edge_forces]
        F += wrenches

    return np.column_stack(F)


def grasp_quality(forces, points, friction_coeffs):
    """Get grasp quality via Ferrari Canny metric, accepting point locations, forces, and friction coefficients as input."""
    if len(forces[0]) != 2: # Dimension of wrench space
        raise ValueError("Force must be 2D or 3D.")

    F = wrench_basis(forces, points, friction_coeffs)
    distance = ferrari_canny(F)
    quality = -1*distance
    return quality

def shape_grasp_quality(shape, theta, b, mu=1.):
    """Get the quality of the grasp on the shape with gripper angle parallel to a ray emitting from the origin with angle `theta`, and linear offset `b` to the left.

    Args:
        shape (shapely.Polygon): shape to grasp
        theta (float): angle of the gripper axis
        b (float): smallest distance from the gripper axis to the origin
        mu (float, optional): Friction coefficient. Defaults to 1.

    Raises:
        ValueError: error if the line does not intersect shape
        QhullError: error if the line intersects shape in a way that results in a deficient geometry, which sometimes happens when the line passes through two adjacent vertices.

    Returns:
        float: Ferrari-Canny grasp metric, the magnitude of the the smallest possible wrench that would escape the gripper, assuming a total contact force across all contacts of 1 unit.
    """
    int_pts = intersection_points(shape, theta, b)
    if len(int_pts)==0: 
        raise ValueError("Line does not intersect shape.")
    vert_pairs = [vertices_from_point(shape, p) for p in int_pts]
    normals = [normal_from_vertices(vert_pair) for vert_pair in vert_pairs]

    forces = list(normals)
    points = list(int_pts)
    friction_coeffs = [mu, mu]
    return grasp_quality(forces, points, friction_coeffs)

def plot_grasp(shape, theta, b, mu=1):
    """Make two subplots: contact points on the polygon, and grasp wrench hull with smallest ball."""

    int_pts = intersection_points(shape, theta, b)
    vert_pairs = [vertices_from_point(shape, p) for p in int_pts]
    normals = [normal_from_vertices(vert_pair) for vert_pair in vert_pairs]

    forces = list(normals)
    points = list(int_pts)
    friction_coeffs = [mu, mu]
    
    fig = plt.figure(figsize=(8,8))
    fig.set_facecolor('grey')
    ax1 = fig.add_subplot(212, projection="3d")
    plot_grasp_quality(forces, points, friction_coeffs, ax1)

    ax2 = fig.add_subplot(211)
    plot_friction_cones_(forces, points, friction_coeffs, ax2)
    plot_shape(shape, ax2)
    ax2.set_aspect('equal')
    plt.tight_layout()

def plot_friction_cones(shape, theta, b, ax, mu=1):
    """Plot the shape, with the contacts and friction cones resulting from a grasp with angle `theta` and offset `b`."""
    int_pts = intersection_points(shape, theta, b)
    vert_pairs = [vertices_from_point(shape, p) for p in int_pts]
    normals = [normal_from_vertices(vert_pair) for vert_pair in vert_pairs]

    forces = list(normals)
    points = list(int_pts)
    friction_coeffs = [mu, mu]

    plot_shape(shape, ax)
    plot_friction_cones_(forces, points, friction_coeffs, ax)



def plot_friction_cones_(forces, points, friction_coeffs, ax):
    """Plot the contact points, with their small friction cones in orange."""

    ax.set_title('Applied Force Position and Friction Cones')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    us = [force[0] for force in forces]
    vs = [force[1] for force in forces]
    ax.scatter(xs, ys)
    ax.quiver(xs, ys, us, vs)

    for f, p, mu in zip(forces, points, friction_coeffs):
        edge_forces = cone_edges(f, mu)
        for edge_force in edge_forces:
            ax.quiver(p[0], p[1], edge_force[0], edge_force[1], color='orange')

def plot_grasp_quality(forces, points, friction_coeffs, ax):
    """Plot the wrench hull and its largest enclosed zero-centered sphere."""

    F = wrench_basis(forces, points, friction_coeffs)
    quality = grasp_quality(forces, points, friction_coeffs)

    pts = F.T

    hull = ConvexHull(pts)

    # Plot defining corner points
    ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")

    # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

    # Make axis label
    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = quality*np.cos(u)*np.sin(v)
    y = quality*np.sin(u)*np.sin(v)
    z = quality*np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

def intersection_points(shape, theta, b, t=100):
    """Return intersection of shape with line of angle `theta`, offset `b`"""
    left, right = line_endpoints(theta, b, t)

    line = LineString([
        (left[0], left[1]),
        (right[0], right[1])
    ])
    intersection = shape.intersection(line)
    if type(intersection) == MultiLineString:
        segment1 = np.array(intersection[0])[0]
        segment2 = np.array(intersection[-1])[1]
        return np.row_stack([segment1, segment2])
    return np.array(shape.intersection(line))

def line_endpoints(theta, b, t=100):
    """ Find endpoints of line parallel to a ray angle theta, with offset `b` to the left of the ray."""
    x0 = b * np.array([-np.sin(theta), np.cos(theta)])
    along = np.array([np.cos(theta), np.sin(theta)])
    return x0+t*along, x0-t*along

def vertices_from_point(polygon, point):
    """Return which two polygon vertices form an edge containing this point"""
    pt = Point(point)
    polin = LineString(list(polygon.exterior.coords))
    points = list(polin.coords)
    for i,j in zip(points, points[1:]):
        if LineString((i,j)).distance(pt) < 1e-8:
            return np.array([i, j])

def normal_from_vertices(point_pair):
    """Return slope of inward normal vector given two (consecutive, clockwise) points from origin"""
    delta_y = point_pair[1,1] - point_pair[0,1]
    delta_x = point_pair[1,0] - point_pair[0,0]
    normal = np.array([-delta_y, delta_x])
    return normal / np.linalg.norm(normal)