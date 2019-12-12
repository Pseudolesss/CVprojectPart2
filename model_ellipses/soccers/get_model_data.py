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


def get_model_data_soccer_ellipse():
    dim = (320, 180)
    image_names = []  # All the images
    images_list = []  # All the images
    images_list_restr = []  # Contain the list of all images except the ones with 0 annotation
    annotations_list_restr = []  # Contain the annotation list for all images except the ones with 0 annotation
    annotations_dict = extract_annotations_soccer()
    annotations_number = []  # All the images

    result = list(Path("../../images_database/soccer/newLabSoustraction/").glob('elps*'))
    for file in result:  # fileName
        image_name = file.name
        image = cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE)
        # We reduce the image size, need to know in what proportion to modify annotation
        image_shape = np.shape(image)
        reduce_coeff = int(round(image_shape[1] / dim[0]))  # 4 if 720p, 6 if 1080p
        image = cv2.resize(image, dsize=dim, interpolation=cv2.INTER_AREA)
        images_list.append(image)
        image_names.append(image_name)

        if image_name in annotations_dict:
            images_list_restr.append(image)
            # select only the biggest annotation
            max_size = 0
            biggest_annotation = []
            for annotation in annotations_dict[image_name]:
                # convert to the good size, cv2 representation (change ymin and ymax) (dim[1] is the image high)
                annotation = convert_annotation(annotation, dim, reduce_coeff)
                size = (annotation[2] - annotation[0]) * (annotation[3] - annotation[1])
                if size > max_size:
                    max_size = size
                    biggest_annotation = annotation
            annotations_list_restr.append(biggest_annotation)
            annotations_number.append(len(annotations_dict[image_name]))
        else:
            annotations_number.append(0)

    return images_list, images_list_restr, annotations_number, annotations_list_restr, annotations_dict


def convert_annotation(annotation, dim, reduce_coeff):
    annotation = list(np.array(annotation) / reduce_coeff)
    annotation[3] = dim[1] - annotation[3]  # New ymin
    annotation[1] = dim[1] - annotation[1]  # New ymax
    tmp = annotation[1]
    annotation[1] = annotation[3]
    annotation[3] = tmp
    return annotation

def get_model_data_soccer_no_ellipse():
    dim = (320, 180)
    images_list = []
    result = list(Path("../../images_database/soccer/noEllipses/").glob('noelps_soccer*'))

    for file in result:  # fileName
        images_list.append(
            cv2.resize(cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE), dsize=dim, interpolation=cv2.INTER_AREA))
    return images_list


if __name__ == '__main__':
    # Test the resize of the image
    get_model_data_soccer_ellipse()
    dim = (320, 180)
    test_image_name = "elps_soccer01_1266.png"
    test_image = cv2.resize(
        cv2.imread("../../images_database/soccer/newLabSoustraction/" + test_image_name, cv2.IMREAD_GRAYSCALE),
        dsize=dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Test', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
