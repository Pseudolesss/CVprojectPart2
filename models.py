from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adadelta
from metrics import dist


def define_custom_loss(weight=None):
    if weight is None:
        weight = [1 / 2, 1 / 2, 1 / 5, 1 / 3, 1 / 3]

    def custom_loss(y_true, y_pred):
        return dist(y_true, y_pred, weight)
    return custom_loss


def create_model_classification_eye(modelName, img_height, img_width):
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
    loss = define_custom_loss()
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def create_model_regression_eye(modelName, img_height, img_width):
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

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Layers for fully connected network and connect it to boolean output
    model.add(Flatten())
    model.add(Dense(5, activation="linear"))

    # optimizer
    # learning rate, momentum to pass over local extrema
    # opt = SGD(lr=0.01, momentum=0.9)
    opt = Adadelta()

    # compile model
    model.compile(loss="mean_squared_error", optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return model
