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
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from model_ellipses.soccers.get_model_data import get_model_data_soccer_ellipse, get_model_data_soccer_no_ellipse
from models import create_model_classification_soccer
from sklearn.model_selection import train_test_split


# Will move percentage of the images from a folder to another
# It will be used to create our Test Set for classification
def move_percentage_of_images(sourceFolder, destinationFolder, percentage):
    result = list(Path(sourceFolder).glob('*.png'))
    random.shuffle(result)

    for i in range(int(len(result) * percentage)):  # fileName
        shutil.move(str(result[i].resolve()), destinationFolder)


def trainClassifier(modelName, images_list_soccer, annotations_number_soccer, images_list_soccer_no_elps,
                    annotations_dict_soccer, nb_epochs, batch_size):
    # open session to use GPU for training model
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # tf.keras.backend.set_session(tf.Session(config=config))

    # Eye images dims
    (img_height, img_width) = (180, 320)

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

    model_history = model.fit(X_train, y_train, validation_split=0.2, epochs=nb_epochs,
                              batch_size=batch_size)

    # train_datagen = ImageDataGenerator(rescale=0,
    #                                    shear_range=0,  # cisaillement
    #                                    zoom_range=0.,  # 0.1
    #                                    width_shift_range=0.,
    #                                    # 0.1  # percentage or pixel number
    #                                    height_shift_range=0.,
    #                                    # 0.1  # percentage or pixel number
    #                                    horizontal_flip=False,  # True
    #                                    dtype='uint8',
    #                                    validation_split=0.2)  # set validation split
    #
    # train_generator = train_datagen.flow(x=X_train, y=y_train, batch_size=batch_size, subset='training')
    # validation_generator = train_datagen.flow(x=X_train, y=y_train, batch_size=batch_size, subset='validation')
    #
    # model_history = model.fit_generator(
    #     train_generator,
    #     # steps_per_epoch=train_generator.samples // batch_size,
    #     validation_data=validation_generator,
    #     # validation_steps=validation_generator.samples // batch_size,
    #     epochs=nb_epochs)
    #
    # test_generator = train_datagen.flow(x=X_test, y=y_test, batch_size=batch_size)

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

    # acc = model_history.history['acc']
    # val_acc = model_history.history['val_acc']
    # plt.plot(epochs, acc, 'y', label='Training acc')
    # plt.plot(epochs, val_acc, 'r', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    # evaluate the model

    scores = model.evaluate(X_test, y_test, batch_size=batch_size)
    # scores = model.evaluate_generator(test_generator, steps=test_generator.n)

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
    images_list_soccer, _, annotations_number_soccer, _, annotations_dict_soccer = get_model_data_soccer_ellipse()
    images_list_soccer_no_elps = get_model_data_soccer_no_ellipse()

    nb_epochs = 15
    batch_size = 50
    trainClassifier("ModelName", images_list_soccer, annotations_number_soccer, images_list_soccer_no_elps,
                    annotations_dict_soccer, nb_epochs, batch_size)

    # # load json and create model
    # with open("./modelclass" + str(nb_epochs) + ".json", "r") as json_file:
    #     loaded_model_json = json_file.read()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("model" + str(nb_epochs) + ".h5")
    #
    # # test loaded model on an example
    # loaded_model.compile(loss='binary_crossentropy', optimizer='adadelta',
    #                      metrics=['accuracy'])
    # example_image = cv2.imread(
    #     "../../images_database/eyes/partial/elps_eye01_2014-11-26_08-49-31-060.png",
    #     cv2.IMREAD_GRAYSCALE)
    # print(loaded_model.predict(np.reshape(example_image, [1, 180, 320, 1])))
