from __future__ import print_function
import random
import numpy as np
import time
from PIL import Image
from skimage import color, io
import matplotlib.pyplot as plt
from typing import Tuple
from skimage import filters
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """
    T = np.eye(4)

    # YOUR CODE HERE
    theta = np.radians(45+90) #manually put in angle

    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    t = np.array([0,0,d]).reshape(3,1)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    # END YOUR CODE

    assert T.shape == (4, 4)
    return T

def apply_transform(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)

    # You'll replace this!
    points_transformed = np.zeros((3, N))

    # YOUR CODE HERE

    points_hom = np.append(points, np.ones([1,N]), axis = 0)
    assert points_hom.shape == (4, N)

    points_transformed = T @ points_hom
    points_transformed = points_transformed[:3,:]
    # END YOUR CODE

    assert points_transformed.shape == (3, N)
    return points_transformed


def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray:
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == float

    # Intersection point between lines
    out = np.zeros(2)

    # YOUR CODE HERE
    a_0 = np.append(a_0, 1) #convert to homogenous
    a_1 = np.append(a_1, 1)
    b_0 = np.append(b_0, 1)
    b_1 = np.append(b_1, 1)

    d_a = np.cross(a_0, a_1) #obtain lines
    d_b = np.cross(b_0, b_1)

    # Convert back to Cartesian by dividing by w    
    out = np.cross(d_a, d_b)[:2] / np.cross(d_a, d_b)[2]
    # END YOUR CODE

    assert out.shape == (2,)
    assert out.dtype == float

    return out


def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v2 (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Wrong shape!"

    optical_center = np.zeros(2)

    # YOUR CODE HERE
    ########
    # A = np.array([
    #     [v1[0] - v2[0], v1[1] - v2[1]],
    #     [v0[0] - v2[0], v0[1] - v2[1]]
    # ])
    # b = np.array([
    #     np.dot(v0, v1 - v2),
    #     np.dot(v0, v1) - np.dot(v1, v2)
    # ])
    # optical_center = np.linalg.solve(A, b)
    
    l01 = v1-v0
    l02 = v2-v0
    l12 = v2-v1
    p_01_02 = v0 + (np.dot(l01, l02) / np.dot(l02, l02)) * l02
    p_01_12 = v1 + (np.dot(-l01, l12) / np.dot(l12, l12)) * l12

    optical_center = intersection_from_lines(p_01_02, v1, p_01_12, v0)

    # END YOUR CODE

    assert optical_center.shape == (2,)
    return optical_center

def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    f = None

    # YOUR CODE HERE
    x0, y0 = v0
    x1, y1 = v1
    cx, cy = optical_center
    f = np.sqrt(-((x0 - cx) * (x1 - cx) + (y0 - cy) * (y1 - cy)))

    # END YOUR CODE

    return float(f)

def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """
    f_mm = None

    # YOUR CODE HERE
    image_width_pixels = image_diagonal_pixels / np.sqrt(2)
    sensor_width_mm = sensor_diagonal_mm /np.sqrt(2)

    f_mm = f * sensor_width_mm/image_width_pixels

    # END YOUR CODE

    return f_mm


