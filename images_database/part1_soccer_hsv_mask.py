import cv2
import numpy as np


def cut_hsv(img, h_min=0, h_max=255, s_min=0, s_max=255, v_min=0, v_max=255):
    """
    Filters the image. Keeps only hsv values that are between two thresholds.

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

    # Applying dilatation then erode to reconstitute largest component and separate it from the others
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # erosion then dilatation of non black pixel
    elem2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))  # dilatation then erosion of non black pixel
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, elem)  # cv2.MORPH_CLOSE
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, elem2)  # cv2.MORPH_OPEN


    img_mask = cv2.bitwise_and(img, img, mask=mask)



    return img_mask