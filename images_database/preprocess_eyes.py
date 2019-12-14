import cv2
import os
import numpy as np
from pathlib import Path


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


def applyContrast(grey_image, factor):
    # Increase image contrast by factor
    grey_image = grey_image.astype(int) * factor

    # Set out of bound values (> 255) to 255
    boundTo255(grey_image)

    # Cast int back to uint8
    return np.uint8(grey_image)


def eyeFullPreprocessing(file, destinationFolder):
    imageName = file.name

    # Open image in grey mode
    img = cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE)

    # Normalize image grey values
    img_norm = normalize_image(img)

    # Increase image contrast by a factor 15
    contrast = np.copy(img_norm)
    contrast = applyContrast(contrast, 15)

    # Double contrast before application of filter
    filtered = np.copy(img_norm)
    filtered = applyContrast(filtered, 2)

    # Apply Bilateral Filtering on normalized image
    cv2.bilateralFilter(img_norm, -1, 20, 5, dst=filtered)

    # Apply Canny to get info on eye structure
    edges = cv2.Canny(filtered, 10, 50)

    # Ostu's algorithm thresholding on contrasted image
    ret, img_th = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert pure white and pure black value to be coherent with Canny result
    for i in range(np.shape(img_th)[0]):
        for j in range(np.shape(img_th)[1]):
            if img_th[i, j] == 0:
                img_th[i, j] = 255
            else:
                img_th[i, j] = 0

    # Final result image
    final_result = img_th + edges

    cv2.imwrite(os.path.join(destinationFolder, imageName), final_result)

def img_eye_partial_preprocessing(img):

    # Normalize image grey values
    img_norm = normalize_image(img)

    # Increase image contrast by a factor 15
    contrast = np.copy(img_norm)
    contrast = applyContrast(contrast, 15)

    ret, img_th = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert pure white and pure black value
    for i in range(np.shape(img_th)[0]):
        for j in range(np.shape(img_th)[1]):
            if img_th[i, j] == 0:
                img_th[i, j] = 255
            else:
                img_th[i, j] = 0

    # Closing morphing operation to fullfil captured iris
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # opening op kernel size 5
    img_mask = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, elem)

    img_th = cv2.bitwise_and(img_th, img_th, mask=img_mask)
    return img_th


def eyePartialPreprocessing(file, destinationFolder):
    imageName = file.name

    # Open image in grey mode
    img = cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE)

    img_th = img_eye_partial_preprocessing(img)

    cv2.imwrite(os.path.join(destinationFolder, imageName), img_th)


if __name__ == '__main__':
    applyPreprocessingDB(".", "./eyes/full", eyeFullPreprocessing,
                         "Team*/elps_eye*")  # Call eyes png files from database only
    applyPreprocessingDB(".", "./eyes/partial", eyePartialPreprocessing,
                         "Team*/elps_eye*")  # Call eyes png files from database only
    applyPreprocessingDB(".", "./eyes/noEllipses/partial", eyePartialPreprocessing,
                         "NoEllipses/noelps_eye*")
