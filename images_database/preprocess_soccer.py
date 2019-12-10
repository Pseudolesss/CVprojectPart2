import cv2
import os
import numpy as np
from pathlib import Path
import part1_soccer_hsv_mask as mask
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

    if nb_components == 0:
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

if __name__ == '__main__':

    sourceFolder = "."
    regexNameFile = "Team*/*soccer*"  # All soccer png files
    destinationFolder = "./soccer/newLabSoustraction"

    result = list(Path(sourceFolder).rglob(regexNameFile))

    for file in result:  # fileName
        imageName = file.name

        imgTest = cv2.imread(str(file.resolve()), cv2.IMREAD_COLOR)

        # if imageName != "elps_soccer01_2153.png":
        #     continue



        # HSV mask applied to get mainly the field
        # The return value is an RGB image
        blur = cv2.blur(imgTest, (5, 5))
        hsv = mask.cut_hsv(blur, h_min=30, h_max=70, s_min=0, s_max=255, v_min=0, v_max=255)

        # result converted to greyscale to get an activation mask
        hsv_bw = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

        # Binary threshold [0][1,255]
        activation = 255 * (hsv_bw > 0).astype('uint8')

        # First apply dilatation because in some picture, brown border (dropped by hsv mask) around white lines
        # Horizontal kernel because it involves the vertical line in the middle of the field
        elem = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
        activation = cv2.morphologyEx(activation, cv2.MORPH_DILATE, elem)

        # HSV sends us back the green field part
        # All other 4-way connected component are noises
        activation = getLargestConnectedComponent(activation)
        activation = activation / 255  # np array of 0 or 1

        # Proceed to dilation to recover lost element
        elem = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 22))
        activation = cv2.morphologyEx(activation, cv2.MORPH_DILATE, elem)

        # Isolate white line through median blur on luminance
        # We proceed to OriginalImage - FilteredImage.
        # The difference should contain field lines luminances

        # LAB space L = Grey level
        medianFiltering = cv2.cvtColor(imgTest, cv2.COLOR_BGR2LAB)
        backgroundLuminance = cv2.medianBlur(medianFiltering[:, :, 0], 25)  # median filtering on brightness component

        line = hsv_bw * activation - backgroundLuminance

        # Binary threshold [0][1,255]
        line[line < 0] = 0
        line[line > 0] = 255
        line[:, 0] = 0  # because of median filter, first column corrupt

        line = line.astype("uint8")

        # # remove small noise
        # elem1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # opening op kernel size 1
        # elem2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # opening op kernel size 2
        # img_mask = cv2.morphologyEx(line, cv2.MORPH_OPEN, elem1)
        #
        #
        # final = cv2.bitwise_and(line, line, mask=img_mask)

        # Remove Connected Component of the given size in pixel and connectivity policy
        final = removeKsizeConnectedComponent(line, 10, connectivity=4)
        final[:, 0] = 0  # because of median filter, first column corrupt

        # cv2.imshow('Results', final)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(destinationFolder, imageName), final)
