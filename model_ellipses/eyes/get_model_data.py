import cv2
import csv
import os
import numpy as np
from pathlib import Path

database_directory = os.path.join(os.getcwd(), '../../images_database')
annotationFile = os.path.join(database_directory, 'CV2019_Annots.csv')


def extract_annotations_eye():
    images_annotations = dict()
    with open(annotationFile, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        result = []

        for row in csv_reader:
            # print(row)
            # print(row[0])
            if 'elps_eye' in row[0]:
                result.append([row[1:], row[0]])

        for eye, image_name in result:

            # We assume only one notation
            tmp = np.array(eye[1:], dtype=np.float32)
            points = np.reshape(tmp, (-1, 2))
            # Convert cartesian y to open cv y coordinate
            for elem in points:
                elem[1] = 240 - elem[1]

            ellipse = cv2.fitEllipse(points)

            # We assume only one notation
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            size = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
            angle = int(ellipse[2])

            images_annotations[image_name] = [center[0], center[1], angle,
                                              size[0], size[1]]

        return images_annotations


def get_model_data_eye_ellipse():
    image_names = []
    images_list = []
    result = list(Path("../../images_database/eyes/partial/").glob('*.png'))
    for file in result:  # fileName
        images_list.append(
            cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE))
        image_names.append(file.name)

    image_annotations = extract_annotations_eye()

    annotations_list = []
    for image_name in image_names:
        if image_name in image_annotations:
            annotations_list.append(image_annotations[image_name])
        else:
            annotations_list.append([])

    return images_list, annotations_list


def get_model_data_eye_no_ellipse():
    images_list = []
    result = list(Path("../../images_database/eyes/noEllipses/partial/").glob(
        'noelps_eye*'))

    for file in result:  # fileName
        images_list.append(
            cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE))

    return images_list
