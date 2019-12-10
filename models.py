from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adadelta, RMSprop, Adam, Adagrad
from metrics import dist

# With the normalized version of the custom loss :
# classic : 412, improved ( angle 1/10 and last dense 256) : 308

def define_custom_loss(weight=None):
    if weight is None:
        weight = [1 / 2, 1 / 2, 1 / 10, 1 / 3, 1 / 3]

    def custom_loss(y_true, y_pred):
        return dist(y_true, y_pred, weight)
        # loss = 0
        # x_center_true, y_center_true, angle_true, main_size_true, sub_size_true = y_true
        # x_center_pred, y_center_pred, angle_pred, main_size_pred, sub_size_pred = y_pred
        # loss += (x_center_pred - x_center_true)**2 + (y_center_pred - y_center_true)**2
        # loss += (main_size_pred - main_size_true)**2 + (sub_size_pred - sub_size_true)**2
        # loss += (angle_pred - angle_true)**2 * abs(main_size_pred - sub_size_pred)
        # return loss

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
    opt = Adadelta()

    # compile model
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

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
    """
    Comparison with all other parameters same (10 epochs, first version custom loss, first version model) :
    Adadelta = 1241
    RMSprop = 1750
    Adam = 1211
    Adagrad = 1023
    """
    opt = Adadelta()
    # opt = RMSprop(learning_rate=0.0001, decay=1e-6)
    # opt = Adam()
    # opt = Adagrad()

    # compile model
    loss = define_custom_loss()
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model
