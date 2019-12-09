import cv2
import csv
import os
import math
import numpy as np
from pathlib import Path

database_directory = os.path.join(os.getcwd(), '../../images_database')
annotationFile = os.path.join(database_directory, 'CV2019_Annots.csv')


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

            min_x = +math.inf
            max_x = -math.inf
            min_y = +math.inf
            max_y = -math.inf
            for x, y in points:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                max_x = max(max_x, x)

            if image_name in images_annotations:
                images_annotations[image_name] = images_annotations[
                                                     image_name] + [
                                                     [min_x, min_y, max_x,
                                                      max_y]]
            else:
                images_annotations[image_name] = [[min_x, min_y, max_x, max_y]]

        return images_annotations




def get_model_data_soccer():
    image_names = []
    images_list = []
    result = list(
        Path("../../images_database/soccer/preprocessed1/").glob('elps*'))
    for file in result:  # fileName
        images_list.append(
            cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE))
        image_names.append(file.name)

    annotations_dict = extract_annotations_soccer()

    annotations_list = []
    for image_name in image_names:
        if image_name in annotations_dict:
            # select only the biggest annotation
            max_size = 0
            biggest_annotation = []
            for annotation in annotations_dict[image_name]:
                size = (annotation[2] - annotation[0]) * (
                            annotation[3] - annotation[1])
                if size > max_size:
                    max_size = size
                    biggest_annotation = annotation
            annotations_list.append(biggest_annotation)
        else:
            annotations_list.append([])

    return images_list, annotations_list, annotations_dict
