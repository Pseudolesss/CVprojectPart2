from images_database.preprocess_soccer import img_soccer_preprocessing
from keras.models import model_from_json
from keras.optimizers import SGD, Adadelta, RMSprop, Adam, Adagrad
import numpy as np
from models import define_custom_loss


def predict_image(input_image_path):
    # First step : Preprocess the image
    preprocessed_image = img_soccer_preprocessing(input_image_path)

    # Second step : Classify the image

    # load the model
    with open("../saved_models/model_class.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_class.h5")
    loaded_model.compile(loss="binary_crossentropy", optimizer=Adadelta(), metrics=['accuracy'])
    result = loaded_model.predict(np.reshape(preprocessed_image, [1, 240, 320, 1]))[0]
    number_ellipses = int(not result)
    print(number_ellipses)

    # Third step : Make the multiple regressions on the image
    with open("../saved_models/model_regr.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_regr.h5")
    loss = define_custom_loss()
    loaded_model.compile(loss=loss, optimizer=Adadelta(), metrics=['accuracy'])
    result = loaded_model.predict(np.reshape(preprocessed_image, [1, 180, 320, 1]))[0]
    center = (int(round(result[0][0])), int(round(result[0][1])))
    size = (int(round(result[0][3])), int(round(result[0][4])))
    angle = int(round(result[0][2]))
    color_test_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    cv2.ellipse(color_test_image, center, size, angle, 0, 360, (0, 255, 0), 1)
    cv2.imshow('Ellipse', color_test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(result)


if __name__ == '__main__':
    predict_image("images_database/Team01/elps_eye01_2014-11-26_08-49-31-060.png")