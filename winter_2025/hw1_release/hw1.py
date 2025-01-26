from __future__ import print_function
import random
import numpy as np
import time
from PIL import Image
from skimage import color, io
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE

    

    padded_image = np.pad(image, ((Hk//2, Hk//2), (Wk//2, Wk//2)), mode='constant')


    for i in range(Hi):
        for j in range(Wi):
            for m in range(Hk):
                for n in range(Wk):
                    out[i, j] += kernel[Hk - 1 - m, Wk - 1 - n] * padded_image[i + m, j + n]
    
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.pad(image, ((pad_height,pad_height),(pad_width,pad_width)), mode='constant', constant_values=0)
    ### END YOUR CODE
    return out



def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    flipped_kernel = np.flip(kernel)
    padded_image = zero_pad(image, Hk//2, Wk//2)
    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            region = padded_image[i:i+Hk,j:j+Wk]
            out[i, j] = np.sum(flipped_kernel * region)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    padded_f = zero_pad(f, Hk//2, Wk//2)
    for i in range(Hi):
        for j in range(Wi):
            region = padded_f[i:i+Hk,j:j+Wk]
            out[i, j] = np.sum(g * region)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    mean = np.mean(g)
    g_0mean = g - np.full((Hk,Wk),mean)
    out = np.zeros((Hi, Wi))
    padded_f = zero_pad(f, Hk//2, Wk//2)
    for i in range(Hi):
        for j in range(Wi):
            region = padded_f[i:i+Hk,j:j+Wk]
            out[i, j] = np.sum(g_0mean * region)
    ### END YOUR CODE

    return out


def check_product_on_shelf(shelf, product):
    out = zero_mean_cross_correlation(shelf, product)
    
    # Scale output by the size of the template
    out = out / float(product.shape[0]*product.shape[1])
    
    # Threshold output (this is arbitrary, you would need to tune the threshold for a real application)
    out = out > 0.025
    
    if np.sum(out) > 0:
        print('The product is on the shelf')
    else:
        print('The product is not on the shelf')


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    u_g = np.mean(g)
    g_0mean = g - np.full((Hk,Wk),u_g)
    std_g = np.std(g)

    out = np.zeros((Hi, Wi))
    padded_f = zero_pad(f, Hk//2, Wk//2)
    for i in range(Hi):
        for j in range(Wi):
            region = padded_f[i:i+Hk,j:j+Wk]
            u_f = np.mean(region)
            std_f = np.std(region)

            region_0mean = region - np.full((Hk,Wk),u_f)
            out[i, j] = np.sum((g_0mean/std_g) * (region_0mean/std_f))
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = (size-1)//2

    for i in range(size):
        for j in range(size):
            kernel[i,j] = (1/(2*np.pi*sigma**2)) * np.exp(-((i-k)**2 + (j-k)**2)/(2*sigma**2))
    ### END YOUR CODE

    return kernel

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    flipped_kernel = np.flip(kernel)
    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            region = padded[i:i+Hk,j:j+Wk]
            out[i, j] = np.sum(flipped_kernel * region)
    ### END YOUR CODE

    return out


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    dx_kernel = np.array([[0,0,0],
                 [0.5,0,-0.5],
                 [0,0,0]]
    )
    out = conv(img, dx_kernel)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    dy_kernel = np.array([[0,0.5,0],
                 [0,0,0],
                 [0,-0.5,0]]
    )
    out = conv(img, dy_kernel)
    ### END YOUR CODE

    return out



def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    g_dx = partial_x(img)
    g_dy = partial_y(img)
    
    G = np.sqrt(g_dx**2 + g_dy**2)
    theta = np.degrees(np.arctan2(g_dy, g_dx)) % 360 #modulo to ensure between [0,360) degrees, also use arctan2 for better divide-by-zero behavior

    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE

    theta = np.mod(theta, 360)

    # Pad arrays with zeros
    G_pad = np.pad(G, pad_width=1, mode='constant', constant_values=0)
    theta_pad = np.pad(theta, pad_width=1, mode='constant', constant_values=0)

    for i in range(H):
        for j in range(W):
            angle = theta_pad[i+1,j+1]
            
            if angle == 0 or angle == 180:
                if G_pad[i+1,j+1] >= G_pad[i+1,j+2] and G_pad[i+1,j+1] >= G_pad[i+1,j]:
                    out[i,j] = G_pad[i+1,j+1]
            elif angle == 45 or angle == 225:
                if G_pad[i+1,j+1] >= G_pad[i,j] and G_pad[i+1,j+1] >= G_pad[i+2,j+2]:
                    out[i,j] = G_pad[i+1,j+1]
            elif angle == 90 or angle == 270:
                if G_pad[i+1,j+1] >= G_pad[i,j+1] and G_pad[i+1,j+1] >= G_pad[i+2,j+1]:
                    out[i,j] = G_pad[i+1,j+1]
            elif angle == 135 or angle == 315:
                if G_pad[i+1,j+1] >= G_pad[i,j+2] and G_pad[i+1,j+1] >= G_pad[i+2,j]:
                    out[i,j] = G_pad[i+1,j+1]
    ### END YOUR CODE

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype=bool)

    ### YOUR CODE HERE
    H, W = img.shape

    for i in range(H):
        for j in range(W):
            if img[i,j] >= high:
                strong_edges[i,j] = True
            elif img[i,j] >= low and img[i,j] < high:
                weak_edges[i,j] = True

    ### END YOUR CODE

    return strong_edges, weak_edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size, sigma)

    smoothed = conv_fast(img, kernel)

    G, theta = gradient(smoothed)

    nms = non_maximum_suppression(G, theta)

    strong_edges, weak_edges = double_thresholding(nms, high, low)

    edge = link_edges(strong_edges, weak_edges)

    ### END YOUR CODE

    return edge

