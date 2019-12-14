import cv2
import os
import numpy as np
from pathlib import Path
import images_database.part1_soccer_hsv_mask as mask
import matplotlib.pyplot as plt


def applyPreprocessingDB(sourceFolder, destinationFolder, function, regexNameFile):
    result = list(Path(sourceFolder).rglob(regexNameFile))

    for file in result:  # fileName
        function(file, destinationFolder)


def normalize_image(norm_image):
    norm_image = norm_image.astype(float)
    ret = (norm_image - np.min(norm_image)) / (np.max(norm_image) - np.min(norm_image) + 1) * 2 ** 8
    return np.uint8(ret)


# Assign out of bound values to max value 2^8-1
# Img is considered as a numpy.uint8 of dim 2
def boundTo255(img):
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if img[i, j] >= 2 ** 8:
                img[i, j] = 2 ** 8 - 1


def getLargestConnectedComponent(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, cv2.CC_STAT_AREA]

    if nb_components < 2:
        return image

    # First check if first element is not a black connected component

    itemindex = np.where(output == 0)
    (x, y) = (itemindex[0][0], itemindex[1][0])

    if image[x][y] != 0:
        max_label = 0
        max_size = sizes[0]
    else:
        max_label = 1
        max_size = sizes[1]

    for i in range(1, nb_components):
        if sizes[i] > max_size:

            itemindex = np.where(output == i)
            (x, y) = (itemindex[0][0], itemindex[1][0])

            if image[x][y] != 0:  # Have to check if biggest connected component is not black
                max_label = i
                max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2


def removeKsizeConnectedComponent(image, size, connectivity=4):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=connectivity)
    sizes = stats[:, -1]

    img2 = np.copy(image)

    for i in range(0, nb_components):
        if sizes[i] < size:
            img2[output == i] = 0

    return img2


def applyContrast(grey_image, factor):
    # Increase image contrast by factor
    grey_image = grey_image.astype(int) * factor

    # Set out of bound values (> 255) to 255
    boundTo255(grey_image)

    # Cast int back to uint8
    return np.uint8(grey_image)


def invert_0_255_image(image):
    image = image + 1  # roundabout overflow (0 => 1, 255 => 0)
    return image * 255


# Interessant mais pas pour ce contexte
def morphologicalSkeleton(img):
    image = np.copy(img)
    skel = np.zeros(image.shape, dtype="uint8")

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    done = False

    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        cv2.subtract(image, temp, temp)
        cv2.bitwise_or(skel, temp, skel)
        image = np.copy(eroded)

        done = np.count_nonzero(image) == 0

    return skel

def preprocessSoccerImage(image):

    # HSV mask applied to get mainly the field
    # The return value is an RGB image
    blur = cv2.blur(image, (5, 5))
    hsv = mask.cut_hsv(blur, h_min=30, h_max=70, s_min=0, s_max=255, v_min=0, v_max=255)

    # Representation of the image in the LAB Color space
    LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # result converted to greyscale to get an activation mask
    activation = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

    # Binary threshold [0][1,255]
    activation = 255 * (activation > 0).astype('uint8')

    # First apply dilatation because in some picture, brown border (dropped by hsv mask) around white lines
    # Horizontal kernel because it involves the vertical line in the middle of the field
    elem = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    activation = cv2.morphologyEx(activation, cv2.MORPH_DILATE, elem)

    # HSV sends us back the green field part
    # All other 4-way connected component are noises
    activation = getLargestConnectedComponent(activation).astype("uint8")

    # Recover the pixel corresponding to white but not in green angle (not in h=[30, 70])
    white_spectrum = mask.cut_hsv(blur, h_min=0, h_max=180, s_min=0, s_max=30, v_min=190, v_max=255)
    white_spectrum = cv2.cvtColor(white_spectrum, cv2.COLOR_BGR2GRAY)

    # Binary threshold [0][1,255]
    white_spectrum = 255 * (white_spectrum > 0).astype('uint8')

    activation = cv2.bitwise_or(activation, white_spectrum)
    activation = 255 * (activation > 0).astype('uint8')
    activation = getLargestConnectedComponent(activation).astype("uint8")

    # Proceed to dilation to recover lost elements
    elem = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    activation = cv2.morphologyEx(activation, cv2.MORPH_DILATE, elem)

    # Isolate white line through median blur on luminance
    # We proceed to OriginalImage - FilteredImage.
    # The difference should contain field lines luminances

    # LAB space L = Luminance level
    medianFiltering = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    backgroundLuminance = cv2.medianBlur(medianFiltering[:, :, 0], 25)  # median filtering on brightness component

    # First normalize activation mask to 1
    activation = activation / 255

    # Applied activation mask full blur image
    full_blur_image_bw = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    line = LAB[:, :, 0] * activation - backgroundLuminance

    # Binary threshold [10][11,255]
    # Like this we are able to isolate the biggest gradient in Luminance
    # Which mainly correspond to white lines
    line[line <= 10] = 0
    line[line > 10] = 255

    for i in range(3):
        line[:, i] = 0  # because of median filter, first columns corrupted

    line = line.astype("uint8")

    # Remove Connected Component of the given size in pixel and connectivity policy
    final = removeKsizeConnectedComponent(line, 40, connectivity=4)

    return final

def SoccerPreprocessing(file, destinationFolder):
    imageName = file.name

    imgTest = cv2.imread(str(file.resolve()), cv2.IMREAD_COLOR)

    # if imageName != "elps_soccer01_2153.png":
    #     continue

    final = preprocessSoccerImage(imgTest)

    cv2.imwrite(os.path.join(destinationFolder, imageName), final)


def applyPreprocessing(sourceFolder):

    regexNameFile = "Team*/*soccer*"  # All soccer png files
    destinationFolder = "./soccer/preprocessed1"

    result = list(Path(sourceFolder).rglob(regexNameFile))

    for file in result:  # fileName
        SoccerPreprocessing(file, destinationFolder)

    result = list(Path(sourceFolder).rglob("NoEllipses/*soccer*"))

    for file in result:  # fileName
        SoccerPreprocessing(file, "./soccer/noEllipses")


if __name__ == '__main__':

    applyPreprocessing(".")

