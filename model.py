from keras.engine import  Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPool2D


def basic_cnn(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    x = MaxPool2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    x = MaxPool2D((2, 2))(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=predictions)
