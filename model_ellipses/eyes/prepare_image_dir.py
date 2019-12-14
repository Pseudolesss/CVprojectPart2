# Use only one time. Images are in images_database/eyes/partial and images_database/eyes/noEllipses/partial
# Make copy to the Model_EYES folder, used by the image generators

# Will move percentage of the images from a folder to another
# It will be used to create our Test Set for classification

from pathlib import Path
import random
import shutil


def move_percentage_of_images(sourceFolder, destinationFolder, percentage):
    """
    Move a pourcentage of image in folder into another folder
    :param sourceFolder: source folder
    :param destinationFolder: destination folder
    :param percentage: percentage of images moved
    """
    result = list(Path(sourceFolder).glob('*.png'))
    random.shuffle(result)

    for i in range(int(len(result) * percentage)):  # fileName
        shutil.move(str(result[i].resolve()), destinationFolder)


def copy_percentage_of_images(sourceFolder, destinationFolder, percentage):
    """
    Copy a pourcentage of image in folder into another folder
    :param sourceFolder: source folder
    :param destinationFolder: destination folder
    :param percentage: percentage of images copied
    """
    result = list(Path(sourceFolder).glob('*.png'))

    for i in range(int(len(result) * percentage)):  # fileName
        shutil.copy(str(result[i].resolve()), destinationFolder)


if __name__ == '__main__':

    copy_percentage_of_images(
        "../../images_database/eyes/partial",
        "../../images_database/Model_EYES/classifier/TrainingValidation/ellipse", 1)
    copy_percentage_of_images(
        "../../images_database/eyes/noEllipses/partial",
        "../../images_database/Model_EYES/classifier/TrainingValidation/noEllipse", 1)

    move_percentage_of_images(
        "../../images_database/Model_EYES/classifier/TrainingValidation/ellipse",
        "../../images_database/Model_EYES/classifier/Test/ellipse", 0.2)
    move_percentage_of_images(
        "../../images_database/Model_EYES/classifier/TrainingValidation/noEllipse",
        "../../images_database/Model_EYES/classifier/Test/noEllipse", 0.2)
