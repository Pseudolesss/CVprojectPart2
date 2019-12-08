import random

import cv2
import csv
import os
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

database_directory = os.path.join(os.getcwd(), '../../images_database')
annotationFile = os.path.join(database_directory, 'CV2019_Annots.csv')


def extract_annotations_eye():
    images_annotations = dict()
    with open(annotationFile, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        result = []

        for row in csv_reader:
            # print(row)
            # print(row[0])
            if 'elps_eye' in row[0]:
                result.append([row[1:], row[0]])

        # According to source code, Width >= Height. ret[1] = size Width and Heigth of rectangle, ret[0] = center of mass (height(vers le bas), width) ret[3]= angle in degrees [0 - 180[
        for eye, image_name in result:

            # We assume only one notation
            tmp = np.array(eye[1:], dtype=np.float32)
            points = np.reshape(tmp, (-1, 2))
            # Convert cartesian y to open cv y coordinate
            for elem in points:
                elem[1] = 240 - elem[1]

            ellipse = cv2.fitEllipse(points)

            # We assume only one notation
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            size = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
            angle = int(ellipse[2])

            images_annotations[image_name] = [center[0], center[1], angle, size[0], size[1]]

        return images_annotations


def extract_annotations_soccer():
    images_annotations = dict()
    with open(annotationFile, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        result = []

        for row in csv_reader:
            # print(row)
            # print(row[0])
            if 'elps_soccer' in row[0]:
                result.append([row[1:], row[0]])

        # According to source code, Width >= Height. ret[1] = size Width and Heigth of rectangle, ret[0] = center of mass (height(vers le bas), width) ret[3]= angle in degrees [0 - 180[
        for soccer, image_name in result:

            # We assume only one notation
            tmp = np.array(soccer[1:], dtype=np.float32)
            points = np.reshape(tmp, (-1, 2))
            # Convert cartesian y to open cv y coordinate
            for elem in points:
                elem[1] = 240 - elem[1]

            min_x = +math.inf
            max_x = -math.inf
            min_y = +math.inf
            max_y = -math.inf
            for x, y in points:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                max_x = max(max_x, x)

            if image_name in images_annotations:
                images_annotations[image_name] = images_annotations[
                                                     image_name] + [[min_x, min_y, max_x, max_y]]
            else:
                images_annotations[image_name] = [[min_x, min_y, max_x, max_y]]

        return images_annotations


def get_model_data_eye():
    image_names = []
    images_list = []
    result = list(Path("../../images_database/eyes/partial/").glob('*.png'))
    for file in result:  # fileName
        images_list.append(
            cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE))
        image_names.append(file.name)

    image_annotations = extract_annotations_eye()

    annotations_list = []
    for image_name in image_names:
        if image_name in image_annotations:
            annotations_list.append(image_annotations[image_name])
        else:
            annotations_list.append([])

    return images_list, annotations_list


def get_no_ellipse_eye():
    image_names = []
    images_list = []
    result = list(Path("../../images_database/eyes/noEllipses/partial/").glob('noelps_eye*'))

    for file in result:  # fileName
        images_list.append(
            cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE))

    return images_list


def get_model_data_soccer():
    image_names = []
    images_list = []
    result = list(Path("../../images_database/soccer/preprocessed1/").glob('elps*'))
    for file in result:  # fileName
        images_list.append(
            cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE))
        image_names.append(file.name)

    image_annotations = extract_annotations_soccer()

    annotations_list = []
    for image_name in image_names:
        if image_name in image_annotations:
            # select only the biggest annotation
            max_size = 0
            biggest_annotation = []
            for annotation in image_annotations[image_name]:
                size = (annotation[2] - annotation[0]) * (annotation[3] - annotation[1])
                if size > max_size:
                    max_size = size
                    biggest_annotation = annotation
            annotations_list.append(biggest_annotation)
        else:
            annotations_list.append([])

    return images_list, annotations_list

import shutil

# Will move percentage of the images from a folder to another
# It will be used to create our Test Set for classification
def move_percentage_of_images(sourceFolder, destinationFolder, percentage):
    result = list(Path(sourceFolder).glob('*.png'))

    random.shuffle(result)

    for i in range(int(len(result) * percentage)):  # fileName
        shutil.move(str(result[i].resolve()), destinationFolder)





import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Reshape
from keras.losses import binary_crossentropy
from keras.optimizers import SGD, Adadelta
from keras_preprocessing.image import ImageDataGenerator


def trainClassifier(modelName, images_list_eye, annotations_list_eye, images_list_eye_no_elps, TVT_Ratio):
    # open session to use GPU for training model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    # Eye images dims
    (img_height, img_width) = (240, 320)

    batch_size = 50
    nb_epochs = 10

    train_data_dir = "../../images_database/Model_EYES/classifier/TrainingValidation/"  # relative
    test_data_dir = "../../images_database/Model_EYES/classifier/Test/"  # relative

    # preparing input values (uint8 images) and output values (boolean)
    # We will call an algorithm splitting the Dataset into Training, Validation and Test sets

    # random.shuffle(images_list_eye)
    # random.shuffle(images_list_eye_no_elps)
    #
    # # Set test set
    # X_test = images_list_eye[:(int(len(images_list_eye) * 0.2))]
    # Y_test = (int(len(images_list_eye) * 0.2) ) * [True]
    # X_test.extend(images_list_eye_no_elps[:(int(len(images_list_eye_no_elps) * 0.2) )])
    # Y_test.extend((int(len(images_list_eye_no_elps) * 0.2) ) * [False])
    #
    # TEST = []
    #
    # for i in range(len(X_test)):
    #     TEST.append([X_test[i], Y_test[i]])
    #
    # random.shuffle(TEST)
    #
    # images_list_eye = images_list_eye[(int(len(images_list_eye) * 0.2) ):]
    # images_list_eye_no_elps = images_list_eye_no_elps[(int(len(images_list_eye_no_elps) * 0.2)):]
    #
    # # Set training/validation set
    # X = images_list_eye
    # Y = len(images_list_eye) * [True]
    # X.extend(images_list_eye_no_elps)
    # Y.extend(len(images_list_eye_no_elps) * [False])
    #
    # DATA = []
    #
    # for i in range(len(X)):
    #     DATA.append([X[i], Y[i]])
    #
    # random.shuffle(DATA)
    #
    # chunks = [DATA[x:x + 100] for x in range(0, len(DATA), 100)]

    # Start creating the model
    model = Sequential(name=modelName)

    # resize input image
    # TODO CHECK INPUT REAL SIZE OF IMAGES GENERATED get dims (100, 240, 320, 3) 100=>? 3=>? (samples, height, width, channels) if dataFormat by default

    # input_shape = (img_height, img_width, 1)
    input_shape = (img_height, img_width, 1)

    # nb_filter, kernel sizes, input shape of the model
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))

    # Max pooling to reduce output dimension
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Dropout to reduce overfitting
    model.add(Dropout(0.3))  # Drop 30 % of inputs

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Layers for fully connected network and connect it to boolean output
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    # optimizer
    # learning rate, momentum to pass over local extrema
    # opt = SGD(lr=0.01, momentum=0.9)
    opt = Adadelta()

    # compile model
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

    model.summary()

    # TODO uses keras capability to augment dataset through image generator

    # TODO take into account TVT ratio
    train_datagen = ImageDataGenerator(rescale=0,
                                       shear_range=0,  # cisaillement
                                       zoom_range=0., #0.1
                                       width_shift_range=0., #0.1  # percentage or pixel number
                                       height_shift_range=0., #0.1  # percentage or pixel number
                                       horizontal_flip=False, # True
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
        class_mode='binary',)

    # model_history = model.fit([x[0] for x in DATA], [y[1] for y in DATA], validation_split=0.2, epochs=nb_epochs, batch_size=batch_size)

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

    acc = model_history.history['acc']
    val_acc = model_history.history['val_acc']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # # evaluate the model
    scores = model.evaluate_generator(test_generator, steps=test_generator.n)
    for i in range(len(model.metrics_names)):
        print("%s: %.2f%%" % (model.metrics_names[i], scores[i] * 100))

    # Save model to json and Weights to H5 files

    # serialize model to JSON
    model_json = model.to_json()
    with open("./model" + str(nb_epochs) + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./model" + str(nb_epochs) + ".h5")
    print("Saved model to disk")

    # # fit model
    # history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)
    # # evaluate the model
    # _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    # _, test_acc = model.evaluate(testX, testy, verbose=0)
    # print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # # plot loss during training
    # pyplot.subplot(211)
    # pyplot.title('Loss')
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # # plot accuracy during training
    # pyplot.subplot(212)
    # pyplot.title('Accuracy')
    # pyplot.plot(history.history['accuracy'], label='train')
    # pyplot.plot(history.history['val_accuracy'], label='test')
    # pyplot.legend()
    # pyplot.show()


if __name__ == '__main__':

    # TODO these next call are to be used for having folders for test set generator
    # move_percentage_of_images("/home/pseudoless/Workspace/CVprojectPart2/images_database/Model_EYES/classifier/TrainingValidation/ellipse", "/home/pseudoless/Workspace/CVprojectPart2/images_database/Model_EYES/classifier/Test/ellipse", 0.2)
    # move_percentage_of_images(
    #     "/home/pseudoless/Workspace/CVprojectPart2/images_database/Model_EYES/classifier/TrainingValidation/noEllipse",
    #     "/home/pseudoless/Workspace/CVprojectPart2/images_database/Model_EYES/classifier/Test/noEllipse", 0.2)

    images_list_eye, annotations_list_eye = get_model_data_eye()
    images_list_eye_no_elps = get_no_ellipse_eye()

    # images_list_soccer, annotations_list_soccer = get_model_data_soccer()
    # print(np.shape(images_list_eye), np.shape(annotations_list_eye))
    # print(np.shape(images_list_soccer), np.shape(annotations_list_soccer))

    Train_Validation_Test_Ratio = (0.7, 0.15, 0.15)
    trainClassifier("ModelName", images_list_eye, annotations_list_eye, images_list_eye_no_elps,
                    Train_Validation_Test_Ratio)


