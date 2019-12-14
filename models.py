from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adadelta
import keras.backend as k


# With the normalized version of the custom loss :
# classic : 412, improved ( angle 1/10 and last dense 256) : 308

def dist(y_true, y_pred, weight):
    """
    compute the weighted distance between y_true and y_pred
    :param y_true: true y
    :param y_pred: predicted y
    :param weight: weight
    :return: distance between y_true and y_pred
    """
    # Use keras instead of numpy in order to avoid symbolic / non symbolic conflicts in the custom loss
    diff = k.abs(y_true - y_pred)
    diff = diff * weight
    diff = diff / k.sum(weight)
    diff = k.sum(diff)
    return diff


def define_custom_loss(weight=None):
    """
    define the custom function that will be used by keras
    :param weight: weight
    :return: custom function
    """
    if weight is None:
        weight = [1 / 2, 1 / 2, 1 / 10, 1 / 3, 1 / 3]

    def custom_loss(y_true, y_pred):
        return dist(y_true, y_pred, weight)
    return custom_loss


def create_model_classification_eye(modelName, img_height, img_width):
    """
    Create the model used for the classification of eyes
    :param modelName: name of the model
    :param img_height: height of image
    :param img_width: width of image
    :return: model created
    """
    # Start creating the model
    model = Sequential(name=modelName)

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

    model.add(Dropout(0.3))  # Drop 30 % of inputs

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Layers for fully connected network and connect it to boolean output
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    # optimizer
    opt = Adadelta()

    # compile model
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model

def create_model_classification_soccer(modelName, img_height, img_width):
    """
    Create the model used for the classification of soccer
    :param modelName: name of the model
    :param img_height: height of image
    :param img_width: width of image
    :return: model created
    """
    # Start creating the model
    model = Sequential(name=modelName)

    # input_shape = (img_height, img_width, 1)
    input_shape = (img_height, img_width, 1)

    # nb_filter, kernel sizes, input shape of the model
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))

    # Max pooling to reduce output dimension
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Dropout to reduce overfitting
    model.add(Dropout(0.4))  # Drop 30 % of inputs

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.4))  # Drop 30 % of inputs

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Layers for fully connected network and connect it to boolean output
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))

    model.add(Dropout(0.4))  # Drop 30 % of inputs

    model.add(Dense(4, activation="softmax"))

    # optimizer
    opt = Adadelta()

    # compile model
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def create_model_regression_eye(modelName, img_height, img_width):
    """
    Create the model used for the regression of eyes
    :param modelName: name of the model
    :param img_height: height of image
    :param img_width: width of image
    :return: model created
    """
    # Start creating the model
    model = Sequential(name=modelName)

    # input_shape = (img_height, img_width, 1)
    input_shape = (img_height, img_width, 1)

    # nb_filter, kernel sizes, input shape of the model
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))

    # Max pooling to reduce output dimension
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Dropout to reduce overfitting
    model.add(Dropout(0.4))  # Drop 30 % of inputs

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.4))

    # Layers for fully connected network and connect it to boolean output
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))

    model.add(Dropout(0.4))

    model.add(Dense(5, activation="linear"))

    # optimizer
    opt = Adadelta()

    # compile model
    loss = define_custom_loss()
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def create_model_regression_soccer(modelName, img_height, img_width):
    """
    Create the model used for the regression of soccer
    :param modelName: name of the model
    :param img_height: height of image
    :param img_width: width of image
    :return: model created
    """

    # Start creating the model
    model = Sequential(name=modelName)

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

    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.3))

    # Layers for fully connected network and connect it to boolean output
    model.add(Flatten())

    model.add(Dense(512, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(1024, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(512, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(256, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))

    model.add(Dropout(0.3))

    # model.add(Dense(64, activation="relu"))
    #
    # model.add(Dropout(0.4))

    model.add(Dense(4, activation="linear"))

    # optimizer
    opt = Adadelta()

    # compile model
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model
