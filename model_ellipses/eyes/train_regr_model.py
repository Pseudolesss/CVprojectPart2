import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from model_ellipses.eyes.get_model_data import get_model_data_eye_ellipse
from models import create_model_regression_eye
from sklearn.model_selection import train_test_split
from keras.optimizers import Adadelta
from models import define_custom_loss
from imgTools import display

# Best result : (200) : 311 loss 94 accuracy

def trainRegressor(modelName, images_list_eye, annotations_list_eye, nb_epochs, batch_size):
    """
    Take the data as input, split and correctly reshape it.
    Fit the eye regression model with the data and evaluate.
    Save the trained model into .json and .h5 files
    :param modelName: Name of the model
    :param images_list_eye: np.array of eye images (np.array) which contains ellipse.
    :param annotations_list_eye: np.array of annotations of the corresp. image (np.array with the 5 ellipses parameters)
    :param nb_epochs: number of epochs
    :param batch_size: size of the batch
    """

    # Eye images dims
    (img_height, img_width) = (240, 320)

    # preparing input values (uint8 images) and output values (5 floats)
    X = images_list_eye
    y = annotations_list_eye
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # create and fit the model
    model = create_model_regression_eye(modelName, img_height, img_width)
    model_history = model.fit(X_train, y_train, validation_split=0.2, epochs=nb_epochs,
                              batch_size=batch_size)

    # Plot the loss evolution
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # # evaluate the model
    scores = model.evaluate(X_test, y_test)
    for i in range(len(model.metrics_names)):
        if model.metrics_names[i] == "loss":
            print("%s: %.2f" % (model.metrics_names[i], scores[i]))
        else:
            print("%s: %.2f%%" % (model.metrics_names[i], scores[i] * 100))

    # Save model to json and Weights to H5 files

    # serialize model to JSON
    model_json = model.to_json()
    with open("./model_regr" + str(nb_epochs) + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./model_regr" + str(nb_epochs) + ".h5")
    print("Saved model to disk")


if __name__ == '__main__':

    # Get the data
    images_list_eye, annotations_list_eye, annotations_dict_eye = get_model_data_eye_ellipse()

    # Train the classifier model
    nb_epochs = 200
    batch_size = 50
    trainRegressor("ModelName", images_list_eye, annotations_list_eye, nb_epochs, batch_size)

    # load the model
    with open("./model_regr" + str(nb_epochs) + ".json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_regr" + str(nb_epochs) + ".h5")
    loss = define_custom_loss()
    loaded_model.compile(loss=loss, optimizer=Adadelta(), metrics=['accuracy'])

    # Test the model on an example
    test_image_name = "elps_eye04_2014-12-14_02-19-30-002.png"
    test_image = cv2.imread("../../images_database/eyes/partial/" + test_image_name, cv2.IMREAD_GRAYSCALE)
    result = loaded_model.predict(np.reshape(test_image, [1, 240, 320, 1]))
    color_test_image = cv2.imread("../../images_database/Team01/" + test_image_name, cv2.IMREAD_COLOR)

    center = (int(round(result[0][0])), int(round(result[0][1])))
    size = (int(round(result[0][3])), int(round(result[0][4])))
    angle = int(round(result[0][2]))
    print("obtained ellipse", result)

    cv2.ellipse(color_test_image, center, size, angle, 0, 360, (0, 255, 0), 1)

    correct = annotations_dict_eye[test_image_name]
    center = (int(round(correct[0])), int(round(correct[1])))
    size = (int(round(correct[3])), int(round(correct[4])))
    angle = int(round(correct[2]))
    print("correct ellipse", correct)

    cv2.ellipse(color_test_image, center, size, angle, 0, 360, (0, 0, 255), 1)

    display('Ellipses', color_test_image)

