# conda install pillow

import random
import shutil
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import model_from_json
from keras_preprocessing.image import ImageDataGenerator
from model_ellipses.eyes.get_model_data import get_model_data_eye_ellipse, get_model_data_eye_no_ellipse
from models import create_model_classification_eye
from sklearn.model_selection import train_test_split


def trainClassifier(modelName, images_list_eye, images_list_eye_no_elps, nb_epochs, batch_size):

    # Eye images dims
    (img_height, img_width) = (240, 320)

    train_data_dir = "../../images_database/Model_EYES/classifier/TrainingValidation/"  # relative
    test_data_dir = "../../images_database/Model_EYES/classifier/Test/"  # relative

    model = create_model_classification_eye(modelName, img_height, img_width)

    # preparing input values (uint8 images) and output values (boolean)
    # We will call an algorithm splitting the Dataset into Training, Validation and Test sets

    # X = images_list_eye + images_list_eye_no_elps
    # y = len(images_list_eye) * [True] + len(images_list_eye_no_elps) * [False]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train = np.expand_dims(X_train, axis=3)
    # X_test = np.expand_dims(X_test, axis=3)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    #
    # model_history = model.fit(X_train, y_train, validation_split=0.2, epochs=nb_epochs,
    #                           batch_size=batch_size)

    train_datagen = ImageDataGenerator(rescale=0,
                                       shear_range=0,  # cisaillement
                                       zoom_range=0.,  # 0.1
                                       width_shift_range=0.,
                                       # 0.1  # percentage or pixel number
                                       height_shift_range=0.,
                                       # 0.1  # percentage or pixel number
                                       horizontal_flip=False,  # True
                                       dtype='uint8',
                                       validation_split=0.2)  # set validation split

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode="grayscale",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')  # set as training data

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,  # same directory as training data
        color_mode="grayscale",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation')  # set as validation data

    model_history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=nb_epochs)

    test_generator = train_datagen.flow_from_directory(
        test_data_dir,  # same directory as training data
        color_mode="grayscale",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary', )

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

    # scores = model.evaluate(X_test, y_test, batch_size=batch_size)
    scores = model.evaluate_generator(test_generator, steps=test_generator.n)

    for i in range(len(model.metrics_names)):
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

    images_list_eye, annotations_list_eye, annotations_dict_eye = get_model_data_eye_ellipse()
    images_list_eye_no_elps = get_model_data_eye_no_ellipse()

    nb_epochs = 10
    batch_size = 50
    trainClassifier("ModelName", images_list_eye, images_list_eye_no_elps, nb_epochs, batch_size)

