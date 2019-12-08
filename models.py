from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adadelta


def create_model_classification_eye(modelName, img_height, img_width):
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
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return model
