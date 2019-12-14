import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from model_ellipses.soccers.get_model_data import get_model_data_soccer_ellipse
from models import create_model_regression_soccer
from sklearn.model_selection import train_test_split
from keras.optimizers import Adadelta
from model_ellipses.soccers.get_model_data import convert_annotation

# best model found (200, original + flip + shift down) : loss 657 accuracy 97.5

def trainRegressor(modelName, images_list_soccer, annotations_list_soccer, nb_epochs, batch_size):
    """
    Take the data as input, split and correctly reshape it.
    Fit the eye regression model with the data and evaluate.
    Save the trained model into .json and .h5 files
    :param modelName: Name of the model
    :param images_list_soccer: np.array of eye images (np.array) which contains ellipse.
    :param annotations_list_soccer: np.array of annotations of the
    corresp. image (np.array with the 4 bounding box param.)
    :param nb_epochs: number of epochs
    :param batch_size: size of the batch
    """

    # Eye images dims
    (img_height, img_width) = (180, 320)

    # preparing input values (uint8 images) and output values (4 floats)
    X = images_list_soccer
    y = annotations_list_soccer
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # create and fit the model
    model = create_model_regression_soccer(modelName, img_height, img_width)
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

    # evaluate the model
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
    _, images_list_restr_soccer, _, annotations_list_restr_soccer, annotations_dict = get_model_data_soccer_ellipse(True)

    # Train the classifier model
    nb_epochs = 200
    batch_size = 50
    # trainRegressor("ModelName", images_list_restr_soccer, annotations_list_restr_soccer, nb_epochs, batch_size)

    # load the model
    with open("./model_regr" + str(nb_epochs) + ".json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_regr" + str(nb_epochs) + ".h5")
    loaded_model.compile(loss="mean_squared_error", optimizer=Adadelta(), metrics=['accuracy'])

    # Test the model on an example
    dim = (320, 180)
    test_image_name = "elps_soccer01_1214.png"
    test_image = cv2.imread( "../../images_database/soccer/preprocessed1/" + test_image_name, cv2.IMREAD_GRAYSCALE)
    test_image_shape = np.shape(test_image)
    reduce_coeff = int(round(test_image_shape[1] / dim[0]))
    test_image = cv2.resize(test_image, dsize=dim, interpolation=cv2.INTER_AREA)
    result = loaded_model.predict(np.reshape(test_image, [1, 180, 320, 1]))[0]

    correct = []
    for annotation in annotations_dict[test_image_name]:
        correct = convert_annotation(annotation, dim, reduce_coeff, test_image_name)
        print("real bounding boxes", correct )
    print("bounding box found", result)
    cv2.rectangle(test_image, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), 255, 1)
    cv2.imshow('Rectangle', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

