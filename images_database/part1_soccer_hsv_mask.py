import cv2
import numpy as np


def cut_hsv(img, h_min=0, h_max=179, s_min=0, s_max=255, v_min=0, v_max=255):
    """
    Filters the image. Keeps only hsv values that are between two thresholds.

    The h value is expressed in degrees and goes from 0->360° (representation divided by two to fit in 8 bits in python/openCV)

    :param img: [np.array] The input image.
    :h_min:     [int] Hue min threshold
    :h_max:     [int] Hue max threshold
    :s_min:     [int] Saturation min threshold
    :s_max:     [int] Saturation max threshold
    :v_min:     [int] Value min threshold
    :v_max:     [int] Value max threshold

    :return:            [np.array] the filtered image in greyscale
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array([h_min, s_min, v_min])
    upp = np.array([h_max, s_max, v_max])

    # Applying hsv mask
    mask = cv2.inRange(hsv, low, upp)

    img_mask = cv2.bitwise_and(img, img, mask=mask)



    return img_mask