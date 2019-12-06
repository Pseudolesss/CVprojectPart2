import cv2
import csv
import os
import math
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

        # According to source code, Width >= Height. ret[1] = size Width and Heigth of rectangle, ret[0] = center of mass (height(vers le bas), width) ret[3]= angle in degrees [0 - 180[
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

            images_annotations[image_name] = [center[0], center[1], angle, size[0], size[1]]

        return images_annotations


def extract_annotations_soccer():
    images_annotations = dict()
    with open(annotationFile, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        result = []

        for row in csv_reader:
            # print(row)
            # print(row[0])
            if 'elps_soccer' in row[0]:
                result.append([row[1:], row[0]])

        # According to source code, Width >= Height. ret[1] = size Width and Heigth of rectangle, ret[0] = center of mass (height(vers le bas), width) ret[3]= angle in degrees [0 - 180[
        for soccer, image_name in result:

            # We assume only one notation
            tmp = np.array(soccer[1:], dtype=np.float32)
            points = np.reshape(tmp, (-1, 2))
            # Convert cartesian y to open cv y coordinate
            for elem in points:
                elem[1] = 240 - elem[1]

            min_x = 0
            max_x = math.inf
            min_y = 0
            max_y = math.inf
            for x, y in points:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                max_x = max(max_x, x)

            if image_name in images_annotations:
                images_annotations[image_name] = images_annotations[
                                                     image_name] + [min_x, min_y, max_x, max_y]
            else:
                images_annotations[image_name] = [[min_x, min_y, max_x, max_y]]

        return images_annotations


def get_model_data_eye():
    image_names = []
    images_list = []
    result = list(Path("../../images_database/eyes/partial/").glob('**/*.png'))
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
            print("ERROR, no annotation corresponding to image")

    return images_list, annotations_list


def get_model_data_soccer():
    image_names = []
    images_list = []
    result = list(Path("../../images_database/soccer/partial/").glob('**/*.png'))
    for file in result:  # fileName
        images_list.append(
            cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE))
        image_names.append(file.name)

    image_annotations = extract_annotations_soccer()

    annotations_list = []
    for image_name in image_names:
        if image_name in image_annotations:
            # select only the biggest annotation
            max_size = 0
            biggest_annotation = []
            for annotation in image_annotations[image_name]:
                size = (annotation[2]-annotation[0])*(annotation[3]-annotation[1])
                if size > max_size:
                    max_size = size
                    biggest_annotation = annotation
            annotations_list.append(biggest_annotation)
        else:
            print("ERROR, no annotation corresponding to image")

    return images_list, annotations_list


if __name__ == '__main__':
    images_list_eye, annotations_list_eye = get_model_data_eye()
    images_list_soccer, annotations_list_soccer = get_model_data_soccer()
