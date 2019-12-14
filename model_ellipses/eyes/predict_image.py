from images_database.preprocess_eyes import img_eye_partial_preprocessing
from keras.models import model_from_json
from keras.optimizers import Adadelta
import numpy as np
from models import define_custom_loss
import cv2
from imgTools import display


def predict_image(input_image_path):
    """
    Take as input a path of a png image and send in result the array of ellipses detected ( empty if no detected)
    Print the resulting ellipse
    :param input_image_path: path to an image file
    :return: ellipse detected (np.array of 5 parameters) if any
    """
    # First step : Preprocess the image
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    preprocessed_image = img_eye_partial_preprocessing(input_image)

    # Second step : Classify the image

    # load the model
    with open("./saved_models/model_class.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./saved_models/model_class.h5")
    loaded_model.compile(loss="binary_crossentropy", optimizer=Adadelta(), metrics=['accuracy'])
    result = loaded_model.predict(np.reshape(preprocessed_image, [1, 240, 320, 1]))[0]
    number_ellipses = int(not result)
    print(number_ellipses)

    if number_ellipses:
        # Third step : Make the regression on the image
        with open("./saved_models/model_regr.json", "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./saved_models/model_regr.h5")
        loss = define_custom_loss()
        loaded_model.compile(loss=loss, optimizer=Adadelta(), metrics=['accuracy'])
        result = loaded_model.predict(np.reshape(preprocessed_image, [1, 240, 320, 1]))[0]
        center = (int(round(result[0])), int(round(result[1])))
        size = (int(round(result[3])), int(round(result[4])))
        angle = int(round(result[2]))
        color_test_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        cv2.ellipse(color_test_image, center, size, angle, 0, 360, (0, 255, 0), 1)
        color_test_image = np.concatenate((cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR), color_test_image), axis=1)

        display('Ellipse', color_test_image)
        print(result)
    else:
        result = []
    return number_ellipses, result


if __name__ == '__main__':
    predict_image("../../images_database/Team03/elps_eye10_2015-01-29_09-01-45-005.png")

    # Good results (best model 1) :
    # images_database/Team03/elps_eye10_2015-01-29_09-00-15-007.png
    # images_database/Team03/elps_eye10_2015-01-29_09-01-45-005.png

