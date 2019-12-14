from pickle import dump
import numpy as np
import cv2
from keras_preprocessing.image import ImageDataGenerator
import random
import copy
import csv


def display_two_images(image1, image2):
    """ Utility function to display two images side by side"""
    result = np.concatenate((image1, image2), axis=1)

    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_images_with_new_anotations(image, annotations):
    """ Utility function to display an image with its corresponding ellipses annotations"""

    result = np.copy(image)

    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    annotations = copy.deepcopy(annotations)

    for ellipse in annotations:
        coords = list()
        for i in range(0, len(ellipse), 2):
            coords.append([int(ellipse[i]), int(result.shape[0] - ellipse[i + 1])])

        coords = np.array([coords], np.int32)
        cv2.polylines(result, [coords], True, (0, 0, 255))
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def flip_image_and_annotations(image, arrayAnnotations):
    """
    This function take as input an image and its corresponding ellipses annotations.
    It creates deep copy of them in order to apply a horizontal flip over the image
    and modify the cytomine coordinates according to the modification.
    It returns a 2-tuple composed of the flipped image and the modified annotations
    """

    shape = image.shape

    modified_image = image.copy()
    arrayAnnotations = copy.deepcopy(arrayAnnotations)
    # Apply horizontal flipping
    cv2.flip(modified_image, 1, dst=modified_image)

    for ellipse in arrayAnnotations:
        for i in range(0, len(ellipse), 2):
            ellipse[i] = shape[1] - ellipse[i]

    # # To check in a plot the resulting image and annotation
    # display_images_with_new_anotations(modified_image, arrayAnnotations)

    return (modified_image, arrayAnnotations)


def random_y_offset_image_and_annotations(image, arrayAnnotations, isShiftUp):
    """
    This function take as arguments an image alongside its corresponding cytomine ellipse annotations and
    a boolean indicating that this function will shift the pixel of the input image up or down.
    The function will apply a 5% to 20% offset of the image height according to a uniform probability law.
    It returns the modified image alongside the modified annotations in a 2-tuple
    """

    shape = image.shape

    modified_image = image.copy()
    arrayAnnotations = copy.deepcopy(arrayAnnotations)

    # Convert single channel image to three channel for the Image Generator instance
    modified_image = cv2.merge((modified_image, modified_image, modified_image))

    imageDataGenerator = ImageDataGenerator()

    # We try to not get a to much similar image, we cap the value to be sure to not exclude an ellipse by zooming in
    if isShiftUp:
        sign = 1
    else:
        sign = -1

    # [5%, 20%] y offset
    pixel_y_Offset = random.uniform(0.05, 0.2) * shape[0] * sign

    # Define dictionary enumerating parameters for the Keras image data generator
    params = {
        "tx": pixel_y_Offset
    }

    modified_image = imageDataGenerator.apply_transform(x=modified_image, transform_parameters=params)

    for ellipse in arrayAnnotations:
        for i in range(0, len(ellipse), 2):
            ellipse[i + 1] = ellipse[i + 1] + pixel_y_Offset

    # display_images_with_new_anotations(cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY), arrayAnnotations)

    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)

    return (modified_image, arrayAnnotations, pixel_y_Offset)


def random_zoom_image_and_annotations(image, arrayAnnotations, isZoomIn):
    """
    This function take as arguments an image alongside its corresponding cytomine ellipse annotations and
    a boolean indicating that this function will zoom in or zoom out of the input image.
    The function will "reduce" the size of the image of a factor 1.2 to 2 according to a uniform law if in Zoom Out mode.
    The function wiil "reduce" the size of the image of a factor 0.8 to 0.95 according to a uniform law if in Zoom in mode.
    The latter behavior is a dangerous one because it pushes out the ellipses out of bound. This is why the range is smaller
    than for the Zoom out behavior.
    It returns the modified image alongside the modified annotations in a 2-tuple
    """

    shape = image.shape

    modified_image = image.copy()
    arrayAnnotations = copy.deepcopy(arrayAnnotations)

    # Convert single channel image to three channel for the Image Generator instance
    modified_image = cv2.merge((modified_image, modified_image, modified_image))

    imageDataGenerator = ImageDataGenerator()

    # We try to not get a to much similar image, we cap the value to be sure to not exclude an ellipse by zooming in
    if not isZoomIn:
        scaleFactor = random.uniform(1.2, 2)  # The ellipse will still be in the image
    else:
        scaleFactor = random.uniform(0.8, 0.95)  # Dangerous procedure

    # Define dictionary enumerating parameters for the Keras image data generator

    params = {
        "zx": scaleFactor,
        "zy": scaleFactor
    }

    modified_image = imageDataGenerator.apply_transform(x=modified_image, transform_parameters=params)

    for ellipse in arrayAnnotations:
        for i in range(0, len(ellipse), 2):
            ellipse[i] = shape[1] / 2 + (1 / scaleFactor) * (ellipse[i] - shape[1] / 2)
            ellipse[i + 1] = shape[0] / 2 + (1 / scaleFactor) * (ellipse[i + 1] - shape[0] / 2)

    # display_images_with_new_anotations(cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY), arrayAnnotations)

    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)

    return (modified_image, arrayAnnotations, scaleFactor)


def write_augmented_image(name, destinationFolder, image):
    cv2.imwrite(destinationFolder + name, image)


def generate_full_augmented_soccer_dataset(preprocessImageFolder, saveAugmentedImagesDirectory, AnnotationsCSVPath):
    with open(AnnotationsCSVPath, 'r') as myfile:
        csv_reader = csv.reader(myfile, delimiter=',')

        csv_list = list()

        for row in csv_reader:
            tmp = list()
            tmp.append(row[0])
            tmp.append(list(row[3:]))
            csv_list.append(tmp)

        dict = {}
        for ellipse in csv_list:
            if ellipse[0] in dict.keys():  # If image already in dict, add ellipse
                tmp = list(map(float, ellipse[1]))  # Transform string into float values
                tmp1 = dict[ellipse[0]].copy()
                tmp1.append(tmp)
                dict[ellipse[0]] = tmp1
            else:
                tmp = list(map(float, ellipse[1]))
                tmp1 = list()
                tmp1.append(tmp)
                dict[ellipse[0]] = tmp1

        # Dictionary to get the final annotations of the augmented dataset
        finalDict = {}

        # First lets generate the flip versions
        for imageName in dict.keys():
            image = cv2.imread(preprocessImageFolder + imageName, cv2.IMREAD_UNCHANGED)

            if image is None:
                print("The image " + imageName + " doesn't exits in " + preprocessImageFolder)
                continue

            (augmented_image, augmented_annotation) = flip_image_and_annotations(image, dict[imageName])

            augmented_name = "FLIP_" + imageName
            write_augmented_image(augmented_name, saveAugmentedImagesDirectory, augmented_image)
            finalDict[augmented_name] = augmented_annotation

        # tmp object because dict change size during loop
        tmp = finalDict.copy()
        for flipImageName in tmp.keys():
            image = cv2.imread(saveAugmentedImagesDirectory + flipImageName, cv2.IMREAD_UNCHANGED)

            if image is None:
                print("The image " + flipImageName + " doesn't exits in " + saveAugmentedImagesDirectory)
                continue

            # Produce twice Shifts and Zoom out
            for nb in range(2):
                (augmented_image, augmented_annotation, factor) = random_y_offset_image_and_annotations(image,
                                                                                                        finalDict[
                                                                                                            flipImageName],
                                                                                                        True)
                augmented_name = "SHIFT_UP_" + str(factor) + "_" + flipImageName
                write_augmented_image(augmented_name, saveAugmentedImagesDirectory, augmented_image)
                finalDict[augmented_name] = augmented_annotation

                (augmented_image, augmented_annotation, factor) = random_y_offset_image_and_annotations(image,
                                                                                                        finalDict[
                                                                                                            flipImageName],
                                                                                                        False)
                augmented_name = "SHIFT_DOWN_" + str(factor) + "_" + flipImageName
                write_augmented_image(augmented_name, saveAugmentedImagesDirectory, augmented_image)
                finalDict[augmented_name] = augmented_annotation

                (augmented_image, augmented_annotation, factor) = random_zoom_image_and_annotations(image, finalDict[
                    flipImageName], False)
                augmented_name = "ZOOM_OUT_" + str(int(100 * factor)) + "_" + flipImageName
                write_augmented_image(augmented_name, saveAugmentedImagesDirectory, augmented_image)
                finalDict[augmented_name] = augmented_annotation

            # Once zoom in

            (augmented_image, augmented_annotation, factor) = random_zoom_image_and_annotations(image, finalDict[
                flipImageName], True)
            augmented_name = "ZOOM_IN_" + str(int(100 * factor)) + "_" + flipImageName
            write_augmented_image(augmented_name, saveAugmentedImagesDirectory, augmented_image)
            finalDict[augmented_name] = augmented_annotation

        # Apply preprocess for original version only
        for imageName in dict.keys():
            image = cv2.imread(preprocessImageFolder + imageName, cv2.IMREAD_UNCHANGED)

            if image is None:
                print("The image " + imageName + " doesn't exits in " + saveAugmentedImagesDirectory)
                continue

            # Produce twice Shifts and Zoom out
            for nb in range(2):
                (augmented_image, augmented_annotation, factor) = random_y_offset_image_and_annotations(image,
                                                                                                        dict[
                                                                                                            imageName],
                                                                                                        True)
                augmented_name = "SHIFT_UP_" + str(factor) + "_" + imageName
                write_augmented_image(augmented_name, saveAugmentedImagesDirectory, augmented_image)
                finalDict[augmented_name] = augmented_annotation

                (augmented_image, augmented_annotation, factor) = random_y_offset_image_and_annotations(image,
                                                                                                        dict[
                                                                                                            imageName],
                                                                                                        False)
                augmented_name = "SHIFT_DOWN_" + str(factor) + "_" + imageName
                write_augmented_image(augmented_name, saveAugmentedImagesDirectory, augmented_image)
                finalDict[augmented_name] = augmented_annotation

                (augmented_image, augmented_annotation, factor) = random_zoom_image_and_annotations(image, dict[
                    imageName], False)
                augmented_name = "ZOOM_OUT_" + str(int(100 * factor)) + "_" + imageName
                write_augmented_image(augmented_name, saveAugmentedImagesDirectory, augmented_image)
                finalDict[augmented_name] = augmented_annotation

            # Once zoom in

            (augmented_image, augmented_annotation, factor) = random_zoom_image_and_annotations(image, dict[
                imageName], True)
            augmented_name = "ZOOM_IN_" + str(int(100 * factor)) + "_" + imageName
            write_augmented_image(augmented_name, saveAugmentedImagesDirectory, augmented_image)
            finalDict[augmented_name] = augmented_annotation

        # Save annotations
        dfg = open(saveAugmentedImagesDirectory + "AugmentedAnnotations.pkl", 'wb')
        dump(finalDict, dfg)
        dfg.close()


if __name__ == '__main__':
    print("Uncomment the comment to proceed to a full generation of an augmented dataset for soccer")

    # preprocessImageFolder = "./soccer/preprocessed1/"
    # saveAugmentedImagesDirectory = "./soccer/AugmentedDataset/"
    # annotationsPath = "./CV2019_Annots_ElpsSoccer.csv"
    #
    # generate_full_augmented_soccer_dataset(preprocessImageFolder, saveAugmentedImagesDirectory, annotationsPath)
