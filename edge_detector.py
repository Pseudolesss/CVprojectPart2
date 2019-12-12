# =====================================================================================================================
# Edge detector for the first part
# =====================================================================================================================

import cv2
import numpy as np


def gradientOfBeucher(input_image, k1=5, k2=5):
    """
    TODO
    """
    kernel = np.ones((k1, k2), np.uint8)

    e = cv2.erode(input_image, kernel, borderType=cv2.BORDER_CONSTANT,
                  iterations=1)
    d = cv2.dilate(input_image, kernel, borderType=cv2.BORDER_CONSTANT,
                   iterations=1)
    return d + e


def canny_vanilla(input_img, lo_thresh=40, hi_thresh=220, sobel_size=3):
    """
    Apply the canny method to the image (without any preprocessing)
    :param input_img: [np.array] The  input image.
    :param lo_thresh:   [int] Low Threshold :  Any edges with intensity
                        gradient lower than this value are sure to be non-edges
    :param hi_thresh:   [int] High Threshold : Any edges with intensity
                        gradient more than this value are sure to be edges
    :param sobel_size:  [int] Size of the Sobel kernel used
                        to get first derivative
    :return:            [np.array] the image containing the local edge points
    """
    return cv2.Canny(input_img, lo_thresh, hi_thresh,
                     apertureSize=sobel_size, L2gradient=True)


def canny_gaussian_blur(input_img, lo_thresh=0, hi_thresh=0, sobel_size=3):
    """
    Apply the canny method to the image (with gaussian blur pre-processing)
    :param input_img:         [np.array] The input image.
    :param lo_thresh:   [int] Low Threshold :  Any edges with intensity
                        gradient lower than this value are sure to be non-edges
    :param hi_thresh:   [int] High Threshold : Any edges with intensity
                        gradient more than this value are sure to be edges
    :param sobel_size:  [int] Size of the Sobel kernel used
                        to get first derivative
    :return:            [np.array] the image containing the local edge points
    """

    i_gaus_kernel_size = 5
    img_filt = cv2.GaussianBlur(input_img, (i_gaus_kernel_size,
                                            i_gaus_kernel_size), 0)

    # Divide the size by 2
    i_reduc_factor = 2
    i_start = i_reduc_factor // 2
    img = img_filt[i_start::i_reduc_factor, i_start::i_reduc_factor]

    # If no threshold specified, use the computed median
    if lo_thresh == 0 and hi_thresh == 0:
        # apply automatic Canny edge detection using the computed median
        med = np.median(img)
        sigma = 0.3
        lo_thresh = int(max(0, (1.0 - sigma) * med))
        hi_thresh = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(img, lo_thresh, hi_thresh,
                     apertureSize=sobel_size, L2gradient=True)


def canny_gaussian_blur_downsize(input_img, lo_thresh=0, hi_thresh=0,
                                 sobel_size=3,i_gaus_kernel_size = 3,gauss_center = 5):
    """
    Apply the canny method to the image (with gaussian blur
    and downsizing pre-processing)
    :param input_img:         [np.array] The input image.
    :param lo_thresh:         [int] Low Threshold :  Any edges with intensity
                              gradient lower than this value are sure to be non-edges
    :param hi_thresh:         [int] High Threshold : Any edges with intensity
                              gradient more than this value are sure to be edges
    :param sobel_size:        [int] Size of the Sobel kernel used
                              to get first derivative
    :param i_gaus_kernel_size:[int] kernel size for Gaussian blur
    :param gauss_center:      [int] center of Gaussian blur
    :return:            [np.array] the image containing the local edge points
    """
    # Downsize the image
    lower_img = cv2.pyrDown(input_img)
    
    # Apply gaussian blur
    img_filt = cv2.GaussianBlur(lower_img, (i_gaus_kernel_size,
                                            i_gaus_kernel_size), gauss_center)
    # Upsize the image
    img = cv2.pyrUp(img_filt)

    # Divide the size by 2
    i_reduc_factor = 2
    i_start = i_reduc_factor // 2
    img = img[i_start::i_reduc_factor, i_start::i_reduc_factor]

    # If no threshold specified, use the computed median
    if lo_thresh == 0 and hi_thresh == 0:
        # apply automatic Canny edge detection using the computed median
        med = np.median(img)
        sigma = 0
        lo_thresh = int(max(0, (1.0 - sigma) * med))
        hi_thresh = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(img, lo_thresh, hi_thresh,
                     apertureSize=sobel_size, L2gradient=True)


def canny_median_blur(input_img, lo_thresh=0, hi_thresh=0, sobel_size=3, downsize=True):
    """
    Apply the canny method to the image (with median blur pre-processing)
    :param input_img:         [np.array] The input image in gray (shape = (? x ?))
    :param lo_thresh:   [int] Low Threshold :  Any edges with intensity
                        gradient lower than this value are sure to be non-edges
    :param hi_thresh:   [int] High Threshold : Any edges with intensity
                        gradient more than this value are sure to be edges
    :param sobel_size:  [int] Size of the Sobel kernel used
                        to get first derivative
    :param downsize:   [bool] whether to downsize the image or not
    :return:            [np.array] the image containing the local edge points
    """

    i_gaus_kernel_size = 5
    img_filt = cv2.medianBlur(input_img, i_gaus_kernel_size)

    # Divide the size by 2
    i_reduc_factor = 2
    i_start = i_reduc_factor // 2
    if downsize:
        img = img_filt[i_start::i_reduc_factor, i_start::i_reduc_factor]
    else:
        img = img_filt

    # If no threshold specified, use the computed median
    if lo_thresh == 0 and hi_thresh == 0:
        # apply automatic Canny edge detection using the computed median
        med = np.median(img)
        sigma = 0.3
        lo_thresh = int(max(0, (1.0 - sigma) * med))
        hi_thresh = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(img, lo_thresh, hi_thresh,
                     apertureSize=sobel_size, L2gradient=True)


def nonLinearLaplacian(input_img, kernel_type=cv2.MORPH_RECT, k1=5, k2=5):
    """
    Apply the non linear Laplacian to an image.
    """
    kernel = cv2.getStructuringElement(kernel_type, (k1, k2))
    return cv2.morphologyEx(input_img, cv2.MORPH_GRADIENT, kernel)


def edgesNLL(img):
    """
    Take a gray image and return a gray image of the edges detected using a
    tuned non Linear Laplacian.
    :param img:         [np.array of shape (? x ?)] The  input image.
    :return:            [np.array] the image containing the local edge points
    """
    gradient_k_size = 2
    i_gaus_kernel_size = 7
    i_gaus_sigma = 2
    img_float = img.astype(np.uint8)

    # apply gaussian blur to remove noise
    img_filt = cv2.GaussianBlur(img_float, (i_gaus_kernel_size,
                                            i_gaus_kernel_size), i_gaus_sigma)

    # detect edges
    img_edges = nonLinearLaplacian(img_filt, kernel_type=cv2.MORPH_RECT,
                                   k1=gradient_k_size, k2=gradient_k_size)

    # apply threshold
    med = np.mean(img_edges)
    lo_thresh = int(2.5 * med)
    thresh_value, img_thresh = cv2.threshold(img_edges, lo_thresh, 255,
                                             cv2.THRESH_BINARY)

    return img_thresh

def edgesDetectionFinal(input_img):
    """
    The edge detector chosen finally after comparing the different candidates
    :param input_img: [np.array] The input image
    :return:    [np.array] The image containing the local edge points
    """
    img_edges = canny_median_blur(input_img)
    # img_edges = ed.canny_gaussian_blur_downsize(input_img)
    return img_edges
