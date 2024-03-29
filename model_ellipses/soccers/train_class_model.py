# conda install pillow
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from model_ellipses.soccers.get_model_data import get_model_data_soccer_ellipse, get_model_data_soccer_no_ellipse
from models import create_model_classification_soccer
from sklearn.model_selection import train_test_split
from keras.optimizers import Adadelta


def trainClassifier(modelName, images_list_soccer, annotations_number_soccer, images_list_soccer_no_elps,
                    nb_epochs, batch_size):
    """
    Take the data as input, split and correctly reshape it.
    Fit the soccer classification model with the data and evaluate.
    Save the trained model into .json and .h5 files
    :param modelName: Name of the model
    :param images_list_soccer: np.array of soccer images (np.array) which contains ellipse.
    :param annotations_number_soccer: np.array of number of annotations  (integer)
    :param images_list_soccer_no_elps: np.array of soccer images (np.array) which doesn't contains ellipse.
    :param nb_epochs: number of epochs
    :param batch_size: size of the batch
    """

    # images dims
    (img_height, img_width) = (180, 320)

    # create the model
    model = create_model_classification_soccer(modelName, img_height, img_width)

    # preparing input values (uint8 images) and output values (array of boolean)
    X = images_list_soccer + images_list_soccer_no_elps
    y = annotations_number_soccer + len(images_list_soccer_no_elps) * [0]
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # If don't want to use the generator
    # model_history = model.fit(X_train, y_train, validation_split=0.2, epochs=nb_epochs, batch_size=batch_size)

    # Declare and flow the generators
    train_datagen = ImageDataGenerator(rescale=0.1, shear_range=0.1, zoom_range=0.1, width_shift_range=0.1,
                                       height_shift_range=0.1, horizontal_flip=True, dtype='uint8',
                                       validation_split=0.2)  # set validation split

    train_generator = train_datagen.flow(x=X_train, y=y_train, batch_size=batch_size, subset='training')

    validation_generator = train_datagen.flow(x=X_train, y=y_train, batch_size=batch_size, subset='validation')

    model_history = model.fit_generator(train_generator, validation_data=validation_generator, epochs=nb_epochs)

    test_generator = train_datagen.flow(x=X_test, y=y_test, batch_size=batch_size)

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

    # If don't want to use the generator
    # scores = model.evaluate(X_test, y_test, batch_size=batch_size)

    scores = model.evaluate_generator(test_generator, steps=test_generator.n)

    for i in range(len(model.metrics_names)):
        if model.metrics_names[i] == "loss":
            print("%s: %.2f" % (model.metrics_names[i], scores[i]))
        else:
            print("%s: %.2f%%" % (model.metrics_names[i], scores[i] * 100))

    # Save model to json and Weights to H5 files

    # serialize model to JSON
    model_json = model.to_json()
    with open("./model_class" + str(nb_epochs) + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./model_class" + str(nb_epochs) + ".h5")
    print("Saved model to disk")


if __name__ == '__main__':

    # Get the data
    images_list_soccer, _, annotations_number_soccer, _, annotations_dict_soccer = get_model_data_soccer_ellipse(True)
    images_list_soccer_no_elps = get_model_data_soccer_no_ellipse()

    # Train the classifier model
    nb_epochs = 60
    batch_size = 50
    trainClassifier("ModelName", images_list_soccer, annotations_number_soccer, images_list_soccer_no_elps,
                    nb_epochs, batch_size)

    # load the model
    with open("./saved_models/model_class.json", "r") as json_file:
        loaded_model_json_class = json_file.read()
    loaded_model_class = model_from_json(loaded_model_json_class)
    loaded_model_class.load_weights("./saved_models/model_class.h5")
    loaded_model_class.compile(loss="categorical_crossentropy", optimizer=Adadelta(), metrics=['accuracy'])

    # Test the model on an example
    dim = (320, 180)
    test_image_name = "elps_soccer01_1326.png"
    test_image = cv2.imread("../../images_database/soccer/preprocessed1/" + test_image_name, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(test_image, dsize=dim, interpolation=cv2.INTER_AREA)
    result = loaded_model_class.predict(np.reshape(image, [1, 180, 320, 1]))[0]
    number_ellipses = result.argmax()
    print(result)
