from recognition.model import basic_cnn
from keras.utils import to_categorical
from recognition.Preprocessor import Preprocessor


def train(patch_dim, tuple_size, num_classes, epochs):
    model = basic_cnn((patch_dim, patch_dim, tuple_size), num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    training_preprocessor = Preprocessor(
        total_patches=80000,
        positive_proportion=0.8,
        patch_dim=patch_dim
    )
    training_patches, labels = training_preprocessor.preprocess()

    validation_preprocessor = Preprocessor(
        total_patches=20000,
        positive_proportion=0.5,
        patch_dim=patch_dim
    )
    validation_patches, validation_labels = validation_preprocessor.preprocess()

    one_hot_labels = to_categorical(labels)
    one_hot_validation_labels = to_categorical(validation_labels)

    print("Training model...")

    model.fit(x=training_patches,
              y=one_hot_labels,
              validation_data=(validation_patches, one_hot_validation_labels),
              epochs=epochs)
    model.save('results/model_result.h5')

    print("Model trained")
