from images_database.preprocess_soccer import img_soccer_preprocessing
from keras.models import model_from_json
from keras.optimizers import SGD, Adadelta, RMSprop, Adam, Adagrad
import numpy as np


def predict_image (input_image_path):

    # First step : Preprocess the image
    preprocessed_image = img_soccer_preprocessing(input_image_path)

    # Second step : Classify the image

    # load the model
    with open("../saved_models/model_class.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_class.h5")
    loaded_model.compile(loss="categorical_crossentropy", optimizer=Adadelta(), metrics=['accuracy'])
    result = loaded_model.predict(np.reshape(preprocessed_image, [1, 180, 320, 1]))[0]
    number_ellipses = result.index(1)[0]
    print(number_ellipses)

    # Third step : Make the multiple regressions on the image
    with open("../saved_models/model_regr.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_regr.h5")
    loaded_model.compile(loss="mean_squared_error", optimizer=Adadelta(), metrics=['accuracy'])
    ellipses_detected = []
    image = preprocessed_image
    for i in range(number_ellipses):
        result = loaded_model.predict(np.reshape(image, [1, 180, 320, 1]))[0]
        ellipses_detected.append(result)
        cv2.rectangle(image, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), 0, -1)
        cv2.imshow('Rectangle', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(ellipses_detected)
    

if __name__ == '__main__':
    predict_image("images_database/Team01/elps_soccer01_1131.png")