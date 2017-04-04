#!python
#cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython cimport view
from libc.math cimport sqrt, pow, fabs, INFINITY
from libc.stdlib cimport malloc, free, realloc
from pylab import imshow, imsave
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from scipy.ndimage import filters
from PIL import Image
from scipy import ndimage
from skimage.morphology import erosion, disk
import os

ctypedef np.float64_t DTYPE_t
ctypedef np.int_t DTYPEi_t
ctypedef double (*math)(double, double) nogil
ctypedef int (*boolean)(double, double) nogil

cdef extern from "utils.h":
    inline double add(double x, double y) nogil
    inline double subtract(double x, double y) nogil
    inline double multiply(double x, double y) nogil
    inline double divide(double x, double y) nogil
    inline double absolute(double x, double y) nogil
    inline int equals(double x, double y) nogil
    inline int not_equals(double x, double y) nogil
    inline int greater_than(double x, double y) nogil
    ctypedef struct Coord:
        int x, y
    ctypedef struct Pixel:
        double x, y, z
    Pixel unfilled_pixel
    ctypedef struct Locations:
        int* x
        int* y
        int length

cdef Coord get_first_where(double[:,:] matrix, double val, boolean boo) nogil:
    cdef:
        int i,j
        Coord coord

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if boo(matrix[i,j], val):
                coord.x = i
                coord.y = j
                return coord

    coord.x = -1
    coord.y = -1
    return coord

cdef Locations where(double[:,:] matrix, double val, boolean boo) nogil:
    cdef:
        int i, j, length = 0
        int dim_x = matrix.shape[0], dim_y = matrix.shape[1]
        Locations locations

    locations.x = <int *>malloc(dim_x * dim_y * sizeof(int))
    locations.y = <int *>malloc(dim_x * dim_y * sizeof(int))

    for i in range(dim_x):
        for j in range(dim_y):
            if boo(matrix[i,j], val):
                locations.x[length] = i
                locations.y[length] = j
                length += 1

    locations.length = length
    locations.x = <int *>realloc(locations.x, length * sizeof(int))
    locations.y = <int *>realloc(locations.y, length * sizeof(int))
    return locations

cdef Locations where_3d(double[:,:,:] matrix, int channel_idx, double val, boolean boo) nogil:
    cdef:
        int i, j, length = 0
        int dim_x = matrix.shape[0], dim_y = matrix.shape[1]
        Locations locations

    locations.x = <int *>malloc(dim_x * dim_y * sizeof(int))
    locations.y = <int *>malloc(dim_x * dim_y * sizeof(int))

    for i in range(dim_x):
        for j in range(dim_y):
            if boo(matrix[i,j,channel_idx], val):
                locations.x[length] = i
                locations.y[length] = j
                length += 1

    locations.length = length
    locations.x = <int *>realloc(locations.x, length * sizeof(int))
    locations.y = <int *>realloc(locations.y, length * sizeof(int))
    return locations

cdef inline double matrix_max(double[:,:] matrix) nogil:
    cdef:
        int i,j
        double max_e = -INFINITY
        double curr_val

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            curr_val = matrix[i,j]
            if curr_val >= max_e:
                max_e = curr_val

    return max_e

cdef inline void do_math_matrix(double[:,:] matrix, Locations location, double oper, math do_math) nogil:
    cdef:
        int i
        int *x = location.x
        int *y = location.y
        int curr_x, curr_y

    for i in range(location.length):
        curr_x = x[i]
        curr_y = y[i]
        matrix[curr_x,curr_y] = do_math(matrix[curr_x,curr_y], oper)

cdef inline void do_math_matrix_3d(double[:,:,:] matrix, Locations location, double oper, math do_math) nogil:
    cdef:
        int i, j
        int *x = location.x
        int *y = location.y
        int curr_x, curr_y

    for i in range(location.length):
        curr_x = x[i]
        curr_y = y[i]
        for j in range(3):
            matrix[curr_x,curr_y,j] = do_math(matrix[curr_x,curr_y,j], oper)

cdef inline void do_math_all_matrix(double[:,:] matrix, double oper, math do_math) nogil:
    cdef int i,j

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i,j] = do_math(matrix[i,j], oper)

cdef inline void do_math_all_matrix_3d(double[:,:,:] matrix, double oper, math do_math):
    cdef int i,j,k

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(3):
                matrix[i,j,k] = do_math(matrix[i,j,k], oper)

cdef inline double[:,:] do_math_matrix_pairwise(double[:,:] A, double[:,:] B, math do_math) nogil:
    cdef:
        int i,j
        int dim_x = A.shape[0], dim_y = A.shape[1]
        double[:,:] C

    with gil:
        C = view.array(shape=(dim_x, dim_y), itemsize=sizeof(double), format="d", mode="c", allocate_buffer=True)

    for i in range(dim_x):
        for j in range(dim_y):
            C[i,j] = do_math(A[i,j], B[i,j])

    return C

cdef inline void do_math_matrix_3d_pairwise(double[:,:,:] A, double[:,:,:] B, math do_math) nogil:
    cdef int i,j,k

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(3):
                A[i,j,k] = do_math(A[i,j,k], B[i,j,k])

cdef inline void assign_vals_to_matrix(double[:,:] matrix, Locations location, double val) nogil:
    cdef:
        int i, curr_x, curr_y
        int *x = location.x
        int *y = location.y

    for i in range(location.length):
        matrix[x[i],y[i]] = val

cdef inline void assign_vals_to_matrix_3d(double[:,:,:] matrix, Locations location, Pixel pixel) nogil:
    cdef:
        int i
        int *x = location.x
        int *y = location.y
        int curr_x, curr_y

    for i in range(location.length):
        curr_x = x[i]
        curr_y = y[i]
        matrix[curr_x,curr_y] = pixel.x
        matrix[curr_x,curr_y] = pixel.y
        matrix[curr_x,curr_y] = pixel.z

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
    return img[x-p:x+p+1,y-p:y+p+1]

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
    return img[x-p:x+p+1,y-p:y+p+1,:]

cdef inline void update(int x, int y, double[:,:] confidence, double[:,:] mask, int patch_size) nogil:
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

cdef inline void paste_patch(int x, int y, double[:,:,:] patch, double[:,:,:] img, int patch_size) nogil:
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
        int i,j

    for i in range(x0, x1):
        for j in range(y0, y1):
            for k in range(3):
                img[i,j,k] = patch[i-x0,j-y0,k]

cdef inline double patch_ssd(double[:,:,:] patch_dst, double[:,:,:] patch_src) nogil:
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
        int m = patch_dst.shape[0], n = patch_dst.shape[1]
        double[:,:,:] patch_srcc = patch_src[:m, :n, :] # ensure two patches are of same dimensions
        int i,j,k
        double ssd_sum = 0

    for i in range(m):
        for j in range(n):
            if patch_dst[i,j,0] != unfilled_pixel.x and \
               patch_dst[i,j,1] != unfilled_pixel.y and \
               patch_dst[i,j,2] != unfilled_pixel.z:
                for k in range(3):
                    ssd_sum += pow(patch_dst[i,j,k] - patch_srcc[i,j,k], 2)

    return ssd_sum

cdef inline double[:,:] hypot(double[:,:] dx, double[:,:] dy) nogil:
    cdef:
        int dim_x = dx.shape[0], dim_y = dy.shape[1]
        int i,j
        double[:,:] h

    with gil:
        h = view.array(shape=(dim_x, dim_y), itemsize=sizeof(double), format="d")

    for i in range(dim_x):
        for j in range(dim_y):
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

cdef inline int patch_all_filled(double[:,:,:] patch) nogil:
    cdef int i,j

    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            if patch[i,j,0] == unfilled_pixel.x and \
               patch[i,j,1] == unfilled_pixel.y and \
               patch[i,j,2] == unfilled_pixel.z:
                return 0

    return 1

cdef inline Coord find_max_priority(Locations boundary_points,
                                    double[:,:] confidence,
                                    double[:,:] dx, double[:,:] dy,
                                    double[:,:] nx, double[:,:] ny,
                                    int patch_size, double alpha) nogil:
    """
    Finds the patch centered at pixels along the fill front which has the highest priority value.

    Parameters
    ----------
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
        int initial_x = boundary_points.x[0], initial_y = boundary_points.y[0]
        int x = initial_x, y = initial_y
        float conf = matrix_sum(get_patch(x, y, confidence, patch_size)) / pow(patch_size, 2) # confidence value
        double[:,:] grad = hypot(dx, dy)
        # a gradient has value of 0 on the boundary so get the maximum gradient magnitude in a patch
        double[:,:] grad_patch = get_patch(x, y, grad, patch_size)
        Coord xx_n_yy = get_first_where(grad_patch, matrix_max(grad_patch), &equals)
        int xx = xx_n_yy.x
        int yy = xx_n_yy.y
        float max_gradx = dx[xx,yy], max_grady = dy[xx,yy]
        float Nx = nx[x,y], Ny = ny[x,y]
        float data = fabs(max_gradx * Nx + max_grady * Ny)
        double curr_p, curr_conf, curr_data = 0
        double[:,:] curr_patch
        int i, curr_bd_x, curr_bd_y
        Coord coord

    if (pow(Nx, 2) + pow(Ny, 2)) != 0:
        data /= (pow(Nx, 2) + pow(Ny, 2))
        
    cdef float max_p = conf * (data / alpha) # initial priority value
    do_math_all_matrix(grad_patch, -1, &absolute)

    # iterate through all patches centered at a pixel on the boundary of
    # unfilled region to find the patch with the highest priority value
    for i in range(boundary_points.length):
        curr_bd_x = boundary_points.x[i]
        curr_bd_y = boundary_points.y[i]
        curr_patch = get_patch(curr_bd_x, curr_bd_y, confidence, patch_size)
        curr_conf = matrix_sum(curr_patch) / pow(patch_size, 2) # confidence value
        # a gradient has value of 0 on the boundary;
        # so get the maximum gradient magnitude in a patch
        grad_patch = get_patch(curr_bd_x, curr_bd_y, grad, patch_size)
        do_math_all_matrix(grad_patch, -1, &absolute)
        xx_n_yy = get_first_where(grad_patch, matrix_max(grad_patch), &equals)
        xx = xx_n_yy.x
        yy = xx_n_yy.y

        max_gradx = dx[xx,yy]
        max_grady = dy[xx,yy]

        Nx = nx[curr_bd_x,curr_bd_y]
        Ny = ny[curr_bd_x,curr_bd_y]

        curr_data = fabs(max_gradx * Nx + max_grady * Ny)
        if (pow(Nx, 2) + pow(Ny, 2)) != 0:
            curr_data /= sqrt(pow(Nx, 2) + pow(Ny, 2))

        curr_p = curr_conf * (curr_data / alpha)
        if curr_p > max_p:
            max_p = curr_p
            x = curr_bd_x
            y = curr_bd_y

    coord.x = x
    coord.y = y

    return coord

cdef inline double[:,:,:] find_exemplar_patch_ssd(double[:,:,:] img, double[:,:,:] patch,
                                                  int x, int y, int patch_size) nogil:
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
        int x_boundary = img.shape[0], y_boundary = img.shape[1]
        # offset the borders of the image by offset pixels to avoid
        # looking through patches that are out of the region
        double[:,:,:] img_copy = img[offset:x_boundary-offset+1, offset:y_boundary-offset+1, :]
        # locations of the unfilled region
        Locations filled_region = where_3d(img_copy, 1, unfilled_pixel.y, &not_equals)
        int* xx = filled_region.x # x coordinates of the unfilled region
        int* yy = filled_region.y # y coordinates of the unfilled region
        double[:,:,:] best_patch, exemplar_patch
        double ssd, min_ssd = INFINITY
        int i, best_x, best_y, x_offset, y_offset

    for i in range(filled_region.length):
        x_offset = xx[i] + offset
        y_offset = yy[i] + offset
        exemplar_patch = get_patch_3d(x_offset, y_offset, img, patch_size)
        if exemplar_patch.shape[0] == patch_size and \
           exemplar_patch.shape[1] == patch_size and \
           patch_all_filled(exemplar_patch) == 1:
            # check if we're not getting the same patch from the parameters
            if x_offset != x and y_offset != y:
                ssd = patch_ssd(patch, exemplar_patch)
                if ssd < min_ssd:
                    best_patch = exemplar_patch
                    best_x = x_offset
                    best_y = y_offset
                    min_ssd = ssd

    return best_patch

cpdef void inpaint(np.ndarray[DTYPE_t, ndim=3] src_im, np.ndarray[DTYPE_t, ndim=2] mask_im,
                   save_name,
                   int gaussian_blur, double gaussian_sigma, int patch_size):
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
        double[:,:,:] src = src_im / 255.0
        double[:,:] mask = mask_im / 255.0
        double[:,:,:] best_patch, max_patch
        double[:,:] confidence = np.zeros((mask.shape[0], mask.shape[1]))
        double[:,:] nx, ny, dx, dy, fill_front
        double[:,:] init_dx, init_dy
        np.ndarray[DTYPE_t, ndim=2] grayscale
        int max_x, max_y
        Coord highest_priority
        Locations boundary_points
        Locations filled_region = where(mask, 0, &not_equals)
        Locations unfilled_region = where(mask, 0, &equals)

    grayscale = src_im[:,:,0] * 0.2125 + src_im[:,:,1] * 0.7154 + src_im[:,:,2] * 0.0721
    grayscale /= 255.0

    # initialize confidence
    assign_vals_to_matrix(confidence, filled_region, 1)

    # place holder value for unfilled pixels
    assign_vals_to_matrix_3d(src, unfilled_region, unfilled_pixel)
    free(filled_region.x)
    free(filled_region.y)

    if gaussian_blur == 1:
        grayscale = ndimage.gaussian_filter(grayscale, gaussian_sigma) # gaussian smoothing for computing gradients

    # compute gradients with sobel operators
    init_dx = ndimage.sobel(grayscale, 0)
    init_dy = -ndimage.sobel(grayscale, 1)

    with nogil:
        while unfilled_region.length > 0:
            # boundary of unfilled region
            with gil:
                fill_front = do_math_matrix_pairwise(mask, erosion(mask, disk(1)), &subtract)
            # pixels where the fill front is located
            boundary_points = where(fill_front, 0, &greater_than)

            # mark region to inpaint
            dx = init_dx
            dy = init_dy
            assign_vals_to_matrix(dx, unfilled_region, 0.0)
            assign_vals_to_matrix(dy, unfilled_region, 0.0)
            free(unfilled_region.x)
            free(unfilled_region.y)

            with gil:
                nx = ndimage.sobel(mask, 0)
                ny = -ndimage.sobel(mask, 1)

            highest_priority = find_max_priority(boundary_points, confidence, dy, dx, ny, nx, patch_size, 255.0)
            free(boundary_points.x)
            free(boundary_points.y)
            max_x = highest_priority.x
            max_y = highest_priority.y

            max_patch = get_patch_3d(max_x, max_y, src, patch_size)
            best_patch = find_exemplar_patch_ssd(src, max_patch, max_x, max_y, patch_size)
            paste_patch(max_x, max_y, best_patch, src, patch_size)
            update(max_x, max_y, confidence, mask, patch_size)

            unfilled_region = where(mask, 0, &equals)

    free(unfilled_region.x)
    free(unfilled_region.y)

    # remove padding and save image
    cdef int offset = patch_size / 2, dim_x = src.shape[0], dim_y = src.shape[1]
    inpainted_img = np.asarray(src[offset:dim_x-offset, offset:dim_y-offset,:])
    imsave(save_name, inpainted_img, format="jpg")

    # show the result
    plt.title('Inpainted Image')
    plt.axis('off')
    plt.show(imshow(inpainted_img))

