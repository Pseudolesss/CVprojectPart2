import cv2
import csv
import os
import numpy as np
from pathlib import Path

database_directory = os.path.join(os.getcwd(), '../../images_database')
annotationFile = os.path.join(database_directory, 'CV2019_Annots.csv')


def extract_annotations_eye():
    """
    extract from the .csv the annotations and put them in a dictionary
    :return: dictionary with image name as key and annotation (np.array with 4 ellipse parameters) as data
    (opencv notation)
    """
    images_annotations = dict()
    with open(annotationFile, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        result = []

        # Take only the ellipses eye rows
        for row in csv_reader:
            if 'elps_eye' in row[0]:
                result.append([row[1:], row[0]])

        for eye, image_name in result:

            tmp = np.array(eye[1:], dtype=np.float32)
            points = np.reshape(tmp, (-1, 2))
            # Convert cyotomine notation to opencv notation
            for elem in points:
                elem[1] = 240 - elem[1]

            ellipse = cv2.fitEllipse(points)

            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            size = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
            angle = int(ellipse[2])

            images_annotations[image_name] = [center[0], center[1], angle, size[0], size[1]]

        return images_annotations


def get_model_data_eye_ellipse():
    """
    get all the interesting information about eye with ellipse
    :return: image_list : np.array of eye images (np.array) which contains ellipse.
            annotations_list : np.array of annotations of the corresp. image (np.array with the 5 ellipses parameters)
            annotations_dict : dictionary with image name as key and annotation (np.array with 4 ellipse parameters)
            as data
    """
    image_names = []
    images_list = []
    result = list(Path("../../images_database/eyes/partial/").glob('*.png'))
    for file in result:  # fileName
        image = cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE)
        images_list.append(image)
        image_names.append(file.name)

    annotations_dict = extract_annotations_eye()

    annotations_list = []
    for image_name in image_names:
        if image_name in annotations_dict:
            annotations_list.append(annotations_dict[image_name])
        else:
            annotations_list.append([])

    return images_list, annotations_list, annotations_dict


def get_model_data_eye_no_ellipse():
    """
    get all the interesting information about eye without ellipse
    :return: image_list : np.array of eye images (np.array) which contains ellipse.
    """
    images_list = []
    result = list(Path("../../images_database/eyes/noEllipses/partial/").glob(
        'noelps_eye*'))

    for file in result:  # fileName
        images_list.append(
            cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE))

    return images_list
