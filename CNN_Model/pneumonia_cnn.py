#!/usr/bin/env python3
"""
Created on Tue Nov  5 17:45:23 2019

@author: michaelperez

Keras CNN that labels chest X-rays as pneumonia (0) or normal (1). The architecture
follows a public Keras tutorial on CelebA gender classification.

Trains the way the saved 2019 models were trained: one pass of plain SGD over the
750x750 images from the DataPipeline in mini-batches of 24, one model.fit() call per
mini-batch. The script committed in 2019 asked for batches of 32, but the optimizer state
saved with each 2019 model counts 218 updates for its one epoch: the 5,232 training images
in batches of 24, the batch size that the script's earlier versions gave the DataPipeline.

    python CNN_Model/pneumonia_cnn.py --data-dir DIR [--conv-blocks 2|3|4] [--epochs 1] [--out FILE]

load_2019_cnn() reads the .h5 files in ML_Models_Saved, which Keras 3 cannot load directly.
Keras uses the PyTorch backend unless KERAS_BACKEND names another one.
"""

import argparse
import json
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

# Filters of the convolutional blocks; the 2019 models used the first 2, 3 or 4
FILTERS = (16, 32, 64, 128)


def build_model(conv_blocks=2, input_shape=(750, 750, 1)):
    """The 2019 architecture with 2, 3 or 4 convolutional blocks."""
    from keras import Input, models, regularizers
    from keras.layers import (
        BatchNormalization,
        Conv2D,
        Dense,
        GlobalAveragePooling2D,
        MaxPooling2D,
    )

    if conv_blocks not in (2, 3, 4):
        raise ValueError(f"conv_blocks must be 2, 3 or 4, not {conv_blocks!r}")

    # starting point
    my_model = models.Sequential()
    my_model.add(Input(shape=input_shape))

    # convolutional blocks of 16, 32, 64 and 128 filters
    for filters in FILTERS[:conv_blocks]:
        my_model.add(Conv2D(filters, (3, 3), activation="relu", padding="same"))
        my_model.add(MaxPooling2D((2, 2), padding="same"))

    # global average pooling
    my_model.add(GlobalAveragePooling2D())
    # fully connected layer
    my_model.add(Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    my_model.add(BatchNormalization())
    # make predictions
    my_model.add(Dense(1, activation="sigmoid"))
    return my_model


def load_2019_cnn(path):
    """
    Loads a CNN saved by this script in 2019 (Keras 2.3.1 with the TensorFlow backend).

    keras.models.load_model() cannot read these files in Keras 3, so the architecture is
    rebuilt from the model description stored in the file and the weights are loaded into it.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if "model_config" not in f.attrs:
            raise ValueError(f"{path} has no model description; it was not saved with model.save()")
        config = f.attrs["model_config"]
    config = json.loads(config.decode("utf-8") if isinstance(config, bytes) else config)
    layers = config["config"]["layers"] if isinstance(config["config"], dict) else config["config"]
    classes = [layer["class_name"] for layer in layers]
    conv_blocks = classes.count("Conv2D")
    expected = ["Conv2D", "MaxPooling2D"] * conv_blocks + [
        "GlobalAveragePooling2D",
        "Dense",
        "BatchNormalization",
        "Dense",
    ]
    if classes != expected or conv_blocks not in (2, 3, 4):
        raise ValueError(
            f"{path} does not hold a 2019 Project Marrow CNN; its layers are {classes}"
        )
    input_shape = tuple(layers[0]["config"]["batch_input_shape"][1:])

    model = build_model(conv_blocks, input_shape)
    model.load_weights(path)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the CNN as in 2019: plain SGD on the 750x750 images of the DataPipeline."
    )
    parser.add_argument(
        "--data-dir", required=True, help="folder with train_images.pickle and test_images.pickle"
    )
    parser.add_argument(
        "--conv-blocks",
        type=int,
        choices=(2, 3, 4),
        default=2,
        help="number of convolutional blocks (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="passes over the training data (default: 1, as for the saved models)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="mini-batch size (default: 24, as for the saved models)",
    )
    parser.add_argument(
        "--out",
        default="CNNClassifier_1_.keras",
        help="file for the trained model (default: CNNClassifier_1_.keras)",
    )
    args = parser.parse_args(argv)

    from ML_DataPipeline import DataPipeline

    train_path = os.path.join(args.data_dir, "train_images.pickle")
    test_path = os.path.join(args.data_dir, "test_images.pickle")
    epochs = args.epochs
    pipeline = DataPipeline(
        "CNN", batch_size=args.batch_size, train_file_path=train_path, test_file_path=test_path
    )

    # 750x750 for the dataset as PictureProcessor.py writes it
    image_shape = pipeline.get_training_batch(0)[0].shape[1:]
    my_model = build_model(args.conv_blocks, input_shape=image_shape)

    # Show a summary of the model. Check the number of trainable parameters
    my_model.summary()

    # compile model
    my_model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        iterations = pipeline.get_iterations()
        for iters in range(iterations):
            x_train, y_train = pipeline.get_training_batch(iters)
            my_model.fit(x_train, y_train, epochs=1, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    my_model.save(args.out)
    return my_model


if __name__ == "__main__":
    main()
