from images_database.preprocess_soccer import preprocessSoccerImage
from keras.models import model_from_json
from keras.optimizers import SGD, Adadelta, RMSprop, Adam, Adagrad
import numpy as np
import cv2
from imgTools import display


def predict_image (input_image_path):
    """
    Take as input a path of a png image and send in result the array of ellipses detected ( empty if no detected)
    Print the resulting ellipses
    :param input_image_path: path to an image file
    :return: np.array of ellipse detected (np.array of the 4 bounding box param.)
    """
    dim = (320, 180)

    # First step : Preprocess the image
    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    preprocessed_image = preprocessSoccerImage(input_image)

    # Second step : Classify the image

    # load the model
    with open("./saved_models/model_class.json", "r") as json_file:
        loaded_model_json_class = json_file.read()
    loaded_model_class = model_from_json(loaded_model_json_class)
    loaded_model_class.load_weights("./saved_models/model_class.h5")
    loaded_model_class.compile(loss="categorical_crossentropy", optimizer=Adadelta(), metrics=['accuracy'])
    image = cv2.resize(preprocessed_image, dsize=dim, interpolation=cv2.INTER_AREA)
    color_image = cv2.resize(input_image, dsize=dim, interpolation=cv2.INTER_AREA)
    result = loaded_model_class.predict(np.reshape(image, [1, 180, 320, 1]))[0]
    number_ellipses = result.argmax()
    print(result)

    # Third step : Make the multiple regressions on the image
    with open("./saved_models/model_regr.json", "r") as json_file:
        loaded_model_json_regr = json_file.read()
    loaded_model_regr = model_from_json(loaded_model_json_regr)
    loaded_model_regr.load_weights("./saved_models/model_regr.h5")
    loaded_model_regr.compile(loss="mean_squared_error", optimizer=Adadelta(), metrics=['accuracy'])
    ellipses_detected = []
    observation_image = image.copy()
    # Each time a ellipse is predicted, we remove it from the image and we predict the modified image until no more ell.
    thresh = 20
    for i in range(number_ellipses):
        result = loaded_model_regr.predict(np.reshape(image, [1, 180, 320, 1]))[0]
        ellipses_detected.append(result)
        cv2.rectangle(image, (int(result[0])-thresh, int(result[1])-thresh), (int(result[2])+thresh, int(result[3])+thresh), 0, -1)
        cv2.rectangle(color_image, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (255, 0, 0), 1)
        display("Ellipse 1", image, 10, 10)
    print(ellipses_detected)
    display("All ellipses", color_image, 10, 10)
    return ellipses_detected


if __name__ == '__main__':
    predict_image("../../images_database/Team03/elps_soccer01_1411.png")

    # Interesting results :
    # images_database/Team03/elps_soccer01_2226.png
    # images_database/Team03/elps_soccer01_2229.png

    # Bad results :
    # images_database / Team03 / elps_soccer01_1411.png
