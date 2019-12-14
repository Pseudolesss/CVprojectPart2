import cv2
import csv
import os
import math
import numpy as np
from pathlib import Path
from pickle import dump, load

database_directory = os.path.join(os.getcwd(), '../../images_database')
annotationFile = os.path.join(database_directory, 'CV2019_Annots.csv')


def points_to_annotations(annotations_dict, image_name, points):
    """
    Put in image_annotations the 4 bounding box parameters of the corresponding points
    :param annotations_dict: dictionary with image name as key and annotation (np.array with 4 ellipse parameters)
    :param image_name:
    :param points: np.array of points (x,y)
    """
    min_x = +math.inf
    max_x = -math.inf
    min_y = +math.inf
    max_y = -math.inf
    for x, y in points:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        max_x = max(max_x, x)

    if image_name in annotations_dict:
        annotations_dict[image_name] = annotations_dict[image_name] + [[min_x, min_y, max_x, max_y]]
    else:
        annotations_dict[image_name] = [[min_x, min_y, max_x, max_y]]


def extract_annotations_soccer():
    """
    extract from the .csv and .pkl the annotations and put them in a dictionary
    :return: dictionary with image name as key and annotation (np.array with 4 ellipse parameters) as data
    (cyotomine notation if from original .csv, opencv notation if from generated .pkl file)
    """
    annotations_dict = dict()

    # Original annotations (csv file)

    with open(annotationFile, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        result = []
        # Take only the ellipses soccer rows
        for row in csv_reader:
            if 'elps_soccer' in row[0]:
                result.append([row[1:], row[0]])

        for soccer, image_name in result:
            tmp = np.array(soccer[1:], dtype=np.float32)
            points = np.reshape(tmp, (-1, 2))
            points_to_annotations(annotations_dict, image_name, points)

    # Generated annotations (pkl file)
    # This dictionary contains the ellipses in the format of a array [[x1, y1, x1, x2, ...], ]
    aze = open("../../images_database/soccer/AugmentedDataset/AugmentedAnnotations.pkl", 'rb')
    generated_annotations_dict = load(aze)

    for image_name in generated_annotations_dict.keys():
        for ellipse in generated_annotations_dict[image_name]:
            points = []
            for i in range(0, len(ellipse), 2):
                (x, y) = (ellipse[i], ellipse[i+1])
                points += [(x, y)]
            points_to_annotations(annotations_dict, image_name, points)
    aze.close()

    return annotations_dict


def get_model_data_soccer_ellipse(generated=False):
    """
    get all the interesting information about eye from the main source (/preprocessed1)
    :param  generated: True if we want to use generated images, False if not
    :return: image_list : np.array of soccer images (np.array)
            image_list_restr : np.array of soccer images (np.array), only if contains ellipse.
            annotations_number: np.array of number of annotations  (integer)
            annotations_list_restr : np.array of annotations of the corresp. image
            (np.array with the 5 ellipses parameters)
            annotations_dict : dictionary with image name as key and annotation (np.array with 4 ellipse parameters)
            as data
    """
    dim = (320, 180)
    image_names = []  # All the images
    images_list = []  # All the images
    images_list_restr = []  # Contain the list of all images except the ones with 0 annotation
    annotations_list_restr = []  # Contain the annotation list for all images except the ones with 0 annotation
    annotations_number = []  # All the images

    annotations_dict = extract_annotations_soccer()

    # List of all stocked images that will be extracted
    result = list(Path("../../images_database/soccer/preprocessed1/").glob('elps*'))
    if generated:
        result += (list(Path("../../images_database/soccer/AugmentedDataset/").glob('FLIP_elps*')))
        result += (list(Path("../../images_database/soccer/AugmentedDataset/").glob('SHIFT_DOWN_*elps*')))

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
                annotation = convert_annotation(annotation, dim, reduce_coeff, image_name)
                size = (annotation[2] - annotation[0]) * (annotation[3] - annotation[1])
                if size > max_size:
                    max_size = size
                    biggest_annotation = annotation
            annotations_list_restr.append(biggest_annotation)
            annotations_number.append(len(annotations_dict[image_name]))

        else:
            annotations_number.append(0)

    return images_list, images_list_restr, annotations_number, annotations_list_restr, annotations_dict


def convert_annotation(annotation, dim, reduce_coeff, image_name):
    """
    Convert annotation to the new size, opencv notation
    :param annotation: annotation to modify
    :param dim: height of the image
    :param reduce_coeff: coefficient of image resize
    :param image_name: image name
    :return: converted annotation
    """
    annotation = list(np.array(annotation) / reduce_coeff)
    # The generated annotations are already in opencv format, the original are in cyotomine format
    if "FLIP" not in image_name and "SHIFT_DOWN" not in image_name:
        annotation[3] = dim[1] - annotation[3]  # New ymin
        annotation[1] = dim[1] - annotation[1]  # New ymax
        tmp = annotation[1]
        annotation[1] = annotation[3]
        annotation[3] = tmp
    return annotation


def get_model_data_soccer_no_ellipse():
    """
    get all the interesting information about soccer without ellipse (from /noEllipses)
    :return: image_list : np.array of eye images (np.array) which contains ellipse.
    """
    dim = (320, 180)
    images_list = []
    result = list(Path("../../images_database/soccer/noEllipses/").glob('noelps_soccer*'))

    for file in result:  # fileName
        images_list.append(
            cv2.resize(cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE), dsize=dim, interpolation=cv2.INTER_AREA))
    return images_list


if __name__ == '__main__':
    # Test the resize of the image
    dim = (320, 180)
    test_image_name = "elps_soccer01_1266.png"
    test_image = cv2.resize(
        cv2.imread("../../images_database/soccer/preprocessed1/" + test_image_name, cv2.IMREAD_GRAYSCALE),
        dsize=dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Test', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
