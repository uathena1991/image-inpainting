#!python
#cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, abs
from cython.parallel import prange, parallel
from pylab import *
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from scipy.ndimage import filters
from PIL import Image
from scipy import ndimage
from skimage.morphology import erosion, disk
import os

ctypedef np.float64_t DTYPE_t
ctypedef np.int_t DTYPEi_t

cdef inline double[:,:] get_patch(int x, int y, double[:,:] img, int patch_size) nogil:
    """
    Gets the patch centered at x and y in the image img with dimensions patch_size by patch_size.

    Parameters
    ----------
    x : int
        x coordinate of the centre of the patch.
    y : int
        y coordinate of the centre of the patch.
    img: 2-D array
        The image where patch is to be obtained from.
    patch_size : int
        Dimensions of the patch; must be odd.

    Returns
    -------
    patch : 2-D array
        The patch of size patch_size by patch_size centered at (x,y) in img.
    """

    cdef int p = patch_size / 2
    return img[x-p:x+p,y-p:y+p]

cdef inline double[:,:,:] get_patch_3d(int x, int y, double[:,:,:] img, int patch_size) nogil:
    """
    Gets the patch centered at x and y in the image img with dimensions patch_size by patch_size.

    Parameters
    ----------
    x : int
        x coordinate of the centre of the patch.
    y : int
        y coordinate of the centre of the patch.
    img: 3-D array
        The image where patch is to be obtained from.
    patch_size : int
        Dimensions of the patch; must be odd.

    Returns
    -------
    patch : 3-D array
        The patch of size patch_size by patch_size centered at (x,y) in img.
    """

    cdef int p = patch_size / 2
    return img[x-p:x+p,y-p:y+p,:]

cdef inline void update(int x, int y, int[:,:] confidence, int[:,:] mask, int patch_size = 9) nogil:
    """
    Updates the confidence values and mask image for the image to be
    to be inpainted.
    
    Parameters
    ----------
    x : int
        x coordinate of the centre of the patch which has been filled.
    y : int 
        y coordinate of the centre of the patch which has been filled.
    confidence : 2-D array
        2-D array holding confidence values for the image to be inpainted.
    mask : 2-D array
        A binary image specifying regions to be inpainted with a value of 0 
        and 1 elsewhere.
    patch_size : int
        Dimensions of the patch; must be odd.
    """

    cdef:
        int p = patch_size / 2
        int x0 = x-p, x1 = x+p+1
        int y0 = y-p, y1 = y+p+1
        int i, j

    for i in range(x0, x1):
        for j in range(y0, y1):
            mask[i,j] = 1
            confidence[i,j] = 1

cdef inline void paste_patch(int x, int y, double[:,:,:] patch, double[:,:,:] img, int patch_size = 9) nogil:
    """
    Updates the confidence values and mask image for the image to be to be inpainted.

    Parameters
    ----------
    x : int
        x coordinate of the centre of the patch which has been filled.
    y : int
        y coordinate of the centre of the patch which has been filled.
    patch : 3-D array
        The patch which has been filled with information from the exemplar patch
    img : 3-D array
        The target image with the unfilled regions.
    patch_size : int
        Dimensions of the patch; must be odd.
    """

    cdef:
        int p = patch_size / 2
        int x0 = x-p, x1 = x+p+1
        int y0 = y-p, y1 = y+p+1
        int i, j

    for i in range(x0, x1):
        for j in range(y0, y1):
            for k in range(3):
                img[i,j,k] = patch[i-x0,j-y0,k]

cdef inline float patch_ssd(double[:,:,:] patch_dst, double[:,:,:] patch_src) nogil:
    """
    Computes the sum of squared differences between patch_dst and patch_src at every pixel.
    
    Parameters
    ----------
    patch_dst : 3-D array
        The patch with an unfilled region.
    patch_src : 3-D array
        The patch being compared to patch_dst. 
        
    Returns
    -------
    sum : float
        The sum of squared differences value of patch_dst and patch_src.
    """

    cdef:
        int m = patch_dst.shape[0]
        int n = patch_dst.shape[1]
        # ensure two patches are of same dimensions
        double[:,:,:] patch_srcc = patch_src[:m, :n, :]
        int i,j,k
        float ssd_sum = 0

    for i in range(m):
        for j in range(n):
            if patch_dst[i,j,1] != 0.9999:
                for k in range(3):
                    ssd_sum += pow(patch_dst[i,j,k] - patch_srcc[i,j,k], 2)

    return ssd_sum

cdef inline double[:,:] hypot(double[:,:] dx, double[:,:] dy):
    cdef:
        double[:,:] h = np.empty((dim_x, dim_y))
        int i,j

    with nogil:
        for i in range(dx.shape[0]):
            for j in range(dy.shape[1]):
                h[i,j] = sqrt(pow(dx[i,j], 2) + pow(dy[i,j], 2))

    return h

cdef inline double matrix_sum(double[:,:] matrix) nogil:
    cdef:
        int i,j
        double m_sum = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            m_sum += matrix[i,j]

    return m_sum

cdef tuple find_max_priority(int[:] boundary_ptx, int[:] boundary_pty,
                             np.ndarray[DTYPE_t, ndim=2] confidence,
                             np.ndarray[DTYPE_t, ndim=2] dx, np.ndarray[DTYPE_t, ndim=2] dy,
                             double[:,:] nx, double[:,:] ny,
                             int patch_size, double alpha = 255.0):
    """
    Finds the patch centered at pixels along the fill front which has the highest priority value.

    Parameters
    ----------
    boundary_ptx : 1-D array
        An array of x coordinates specifying the locations of the pixels of
        the boundary of the fill region.
    boundary_pty : 1-D array
        An array of y coordinates specifying the locations of the pixels of
        the boundary of the fill region.
    confidence : 2-D array
        2-D array holding confidence values for the image to be inpainted.
    dx : 2-D array
        The gradient image of the unfilled image in x direction.
    dy : 2-D array
        The gradient image of the unfilled image in y direction.
    nx : 2-D array
        The normal image of the mask in x direction.
    ny : 2-D array
        The normal image of the mask in y direction.
    patch_size : int
        Dimensions of the patch; must be odd.
    alpha :
        The normalization factor; suggested value of 255 as described by
        Criminisi.


    Returns
    -------
    max : float
        The highest priority value.
    x : int
        x coordinate of the center of the patch at which the highest priority
        value was computed
    y : int
        y coordinate of the center of the patch at which the highest priority
        value was computed
    """
    # initialize first priority value
    cdef:
        float conf = matrix_sum(get_patch(boundary_ptx[0],
                                      boundary_pty[0], 
                                      confidence, 
                                      patch_size))/pow(patch_size, 2) # confidence value
        double[:,:] grad = hypot(dx, dy)
        # a gradient has value of 0 on the boundary;
        # so get the maximum gradient magnitude in a patch
        np.ndarray[DTYPE_t, ndim=2] grad_patch = np.fabs(get_patch(boundary_ptx[0],
                                                               boundary_pty[0],
                                                               grad,
                                                               patch_size))
        int xx = np.where(grad_patch == np.max(grad_patch))[0][0]
        int yy = np.where(grad_patch == np.max(grad_patch))[1][0]
        float max_gradx = dx[xx,yy]
        float max_grady = dy[xx,yy]
        float Nx = nx[boundary_ptx[0], boundary_pty[0]]
        float Ny = ny[boundary_ptx[0], boundary_pty[0]]
        int x = boundary_ptx[0]
        int y = boundary_pty[0]
        float data = abs(max_gradx * Nx + max_grady * Ny)
        double curr_p, curr_conf, curr_data = 0
        double[:,:] curr_patch

    if (pow(Nx, 2) + pow(Ny, 2)) != 0:
        data /= (pow(Nx, 2) + pow(Ny, 2))
        
    cdef:
        float max_p = conf * (data / alpha) # initial priority value
        int i
        int curr_bd_x, curr_bd_y

    # iterate through all patches centered at a pixel on the boundary of 
    # unfilled region to find the patch with the highest priority value
    for i in range(boundary_ptx.shape[0]):
        curr_patch = get_patch(boundary_ptx[i], boundary_pty[i], confidence, patch_size)
        curr_conf = matrix_sum(curr_patch)/pow(patch_size, 2) # confidence value
        # a gradient has value of 0 on the boundary;
        # so get the maximum gradient magnitude in a patch
        grad_patch = np.fabs(get_patch(boundary_ptx[i], boundary_pty[i], grad, patch_size))
        xx = np.where(grad_patch == np.max(grad_patch))[0,0]
        yy = np.where(grad_patch == np.max(grad_patch))[1,0]

        max_gradx = dx[xx,yy]
        max_grady = dy[xx,yy]

        curr_bd_x = boundary_ptx[i]
        curr_bd_y = boundary_pty[i]
        Nx = nx[curr_bd_x,curr_bd_y]
        Ny = ny[curr_bd_x,curr_bd_y]

        curr_data = abs(max_gradx * Nx + max_grady * Ny)
        if (pow(Nx, 2) + pow(Ny, 2)) != 0:
            curr_data /= sqrt(pow(Nx, 2) + pow(Ny, 2))

        curr_p = curr_conf * (curr_data / alpha)
        if curr_p > max_p:
            max_p = curr_p
            x = boundary_ptx[i]
            y = boundary_pty[i]

    return max_p, x, y
    

    
cdef double[:,:,:] find_exemplar_patch_ssd(double[:,:,:] img, double[:,:,:] patch,
                                   int x, int y, int patch_size = 9):
    """
    Finds the best exemplar patch with the minimum sum of squared differences.
    
    Parameters
    ----------
    img : 3-D array
        The image with unfilled regions to be inpainted.
    patch : 3-D array
        The patch centered at (x, y) with the highest priority value and an 
        unfilled region.
    x : int
        The x coordinate of the center of the patch with tbe highest
        priority value.
    y : int
        The y coordinate of the center of the patch with tbe highest
        priority value.
    patch_size : int
        Dimensions of the patch size; must be odd.
        
    Returns
    -------
    best_patch : 3-D array
        The argmin patch with the lowest ssd with patch.
    best_x : int
        The x coordinate which best_patch is centered at.
    best_y : int
        The y coordinate which best_patch is centered at.
    """
    
    cdef: 
        int offset = patch_size / 2
        int x_boundary = img.shape[0]
        int y_boundary = img.shape[1]
        int i = 0
        # offset the borders of the image by offset pixels to avoid
        # looking through patches that are out of the region
        np.ndarray[DTYPE_t, ndim=3] img_copy = img[offset:x_boundary-offset+1, offset:y_boundary-offset+1]
        
    # locations of the unfilled region
    filled_r = np.where(img_copy[:,:,1] != 0.9999)
    
    cdef:
        int[:] xx = filled_r[0] # x coordinates of the unfilled region
        int[:] yy = filled_r[1] # y coordinates of the unfilled region
        int unfilled_pixels = xx.shape[0]
        np.ndarray[DTYPE_t, ndim=3] exemplar_patch
        np.ndarray[DTYPE_t, ndim=3] best_patch
        int best_x, best_y
        float ssd, min_ssd = np.inf

    for i in range(unfilled_pixels):
        exemplar_patch = get_patch_3d(xx[i] + offset, yy[i] + offset, img, patch_size)
        if (exemplar_patch.shape[0] == patch_size and 
            exemplar_patch.shape[1] == patch_size):
            # check if we're not getting the same patch from the parameters
            # and if the potential exemplar patch has no unfilled regions
            if ((xx[i] + offset) != x and 
                (yy[i] + offset) != y and 
                (exemplar_patch[:,:,1] == 0.9999).any()):
                ssd = patch_ssd(patch, exemplar_patch)
                if ssd < min_ssd:
                    best_patch = exemplar_patch
                    best_x = xx[i] + offset
                    best_y = yy[i] + offset
                    min_ssd = ssd

    return best_patch

cpdef void inpaint(src_im, mask_im, save_name, int gaussian_blur=0, int gaussian_sigma=1, int patch_size=9):
    """
    Runs the inpainting algorithm.
    
    Parameters
    ----------
    src_im : string
        Source image. 3-D array.
    mask_im : string 
        Mask indicating areas in the source image to inpaint. 2-D array.
    save_name : string
        Name of the source image. Will be used to save intermediate and final
        result of the algorithm.
    gaussian_blur : int
        Specifies whether to use Gaussian blur or not; 0 for no, 1 for yes.
    gaussian_sigma: double
        Value for the sigma for Gaussian blur.
    patch_size : int
        Dimensions of the path; must be odd.
    """

    cdef:
        np.ndarray src = src_im
        np.ndarray mask = mask_im
        np.ndarray[DTYPE_t, ndim=3] unfilled_img
        np.ndarray[DTYPE_t, ndim=2] grayscale
        np.ndarray confidence = np.zeros(mask_im.shape)
        np.ndarray [DTYPEi_t, ndim=1] boundary_ptx, boundary_pty
        int max_x, max_y, patch_count = 0
        np.ndarray[DTYPE_t, ndim=3] max_patch, copied_patch
        double[:,:] nx, ny
        np.ndarray fill_front, dx, dy
        double[:,:,:] best_patch

    unfilled_img = src/255.0
    mask = mask/255.0
    grayscale = src[:,:,0]*.2125 + src[:,:,1]*.7154 + src[:,:,2]*.0721
    grayscale /= 255.0
    
    # initialize confidence
    confidence[mask != 0] = 1
    
    # place holder value for unfilled pixels
    unfilled_img[mask == 0.0] = [0.0, 0.9999, 0.0]
        
    if gaussian_blur == 1:
        # gaussian smoothing for computing gradients
        grayscale = ndimage.gaussian_filter(grayscale, gaussian_sigma) 
    
    while (mask == 0)[0].any():
        # boundary of unfilled region
        fill_front = mask - erosion(mask, disk(1)) 
        # pixels where the fill front is located
        boundary_ptx = np.where(fill_front > 0)[0] # x coordinates
        boundary_pty = np.where(fill_front > 0)[1] # y coordinates
        # compute gradients with sobel operators
        dx = ndimage.sobel(grayscale, 0)
        dy = -ndimage.sobel(grayscale, 1)
        # mark region to inpaint
        dx[mask == 0] = 0.0
        dy[mask == 0] = 0.0
        # compute normals
        nx = ndimage.sobel(mask, 0)
        ny = -ndimage.sobel(mask, 1)
        
        highest_priority = find_max_priority(boundary_ptx, boundary_pty,
                                             confidence, 
                                             dy, dx, ny, nx,
                                             patch_size)

        max_x = highest_priority[1]
        max_y = highest_priority[2]
        max_patch = get_patch(max_x, max_y, unfilled_img, patch_size)
        
        best_patch = find_exemplar_patch_ssd(unfilled_img, max_patch, max_x, max_y, patch_size)
        paste_patch(max_x, max_y, best_patch, unfilled_img, patch_size)

        update(max_x, max_y, confidence, mask, patch_size)

        patch_count += 1
        print patch_count, 'patches inpainted'

    imsave(save_name, unfilled_img)
    # show the result
    plt.title('Inpainted Image')
    plt.axis('off')
    plt.show(imshow(unfilled_img))
