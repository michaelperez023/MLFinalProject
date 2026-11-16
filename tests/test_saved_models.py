"""
The models saved in 2019 (ML_Models_Saved/). The 12 SGD pickles load only with scikit-learn
0.23 or older (the requirements-2019.txt environment), the Naive Bayes one also with current
versions; the CNN tests need Keras 3.
"""

import json
import pickle
import pickletools
import re

import numpy as np
import pytest
from helpers import REPO, SAVED_MODELS, run_script, write_pipeline_pickle

SKLEARN_PICKLES = sorted(SAVED_MODELS.glob("*.pickle"))
CNN_FILES = sorted(SAVED_MODELS.glob("*.h5"))


def sklearn_version():
    import sklearn

    return tuple(int(part) for part in re.findall(r"\d+", sklearn.__version__)[:2])


def test_the_saved_files_are_all_there():
    names = [p.stem for p in SKLEARN_PICKLES]
    expected = ["NBClassifier_1"] + [
        f"{kind}Classifier_{epochs}"
        for kind in ("LOG", "PER", "SVM")
        for epochs in (1, 50, 100, 200)
    ]
    assert sorted(names) == sorted(expected)
    assert [p.stem for p in CNN_FILES] == [
        "CNN_2layers_1epoch",
        "CNN_3layers_1epoch",
        "CNN_4layers_1epoch",
    ]


@pytest.mark.parametrize("path", SKLEARN_PICKLES, ids=lambda p: p.stem)
def test_pickles_only_reference_numpy_and_sklearn(path):
    # Unpickling can run code; these files only rebuild numpy arrays and scikit-learn objects
    names, strings = set(), []
    with open(str(path), "rb") as f:
        for opcode, argument, _ in pickletools.genops(f):
            if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                strings.append(argument)
            elif opcode.name == "GLOBAL":
                names.add(argument.split(" ")[0])
            elif opcode.name == "STACK_GLOBAL":
                names.add(strings[-2])
            elif opcode.name == "STOP":
                break
        assert f.read() == b"", "more than one object in the file"
    assert names and all(name.split(".")[0] in ("numpy", "sklearn") for name in names)


@pytest.mark.parametrize("path", SKLEARN_PICKLES, ids=lambda p: p.stem)
def test_2019_sklearn_models_load_and_predict(path):
    if "NB" not in path.stem and sklearn_version() >= (0, 24):
        pytest.skip("the 2019 SGD models need scikit-learn 0.23 or older (requirements-2019.txt)")
    with open(str(path), "rb") as f:
        model = pickle.load(f)
    assert list(model.classes_) == [0, 1]
    weights = model.feature_count_ if "NB" in path.stem else model.coef_
    assert weights.shape[1] == 750 * 750
    features = np.random.default_rng(0).integers(0, 256, size=(3, 750 * 750)).astype(float)
    if "NB" not in path.stem:
        features = (features - features.mean(axis=1, keepdims=True)) / features.std(
            axis=1, keepdims=True
        )
    assert set(model.predict(features)) <= {0, 1}


def test_naive_bayes_training_counts():
    # The 2019 training split: 3,883 pneumonia (label 0) and 1,349 normal (label 1) images
    with open(str(SAVED_MODELS / "NBClassifier_1.pickle"), "rb") as f:
        model = pickle.load(f)
    assert model.class_count_.tolist() == [3883, 1349]


# CNNs ------------------------------------------------------------------------------------------


def keras_or_skip():
    return pytest.importorskip("keras")


@pytest.mark.parametrize("path", CNN_FILES, ids=lambda p: p.stem)
def test_2019_cnn_loads_with_its_saved_weights(path):
    keras_or_skip()
    import h5py
    from pneumonia_cnn import load_2019_cnn

    model = load_2019_cnn(str(path))
    conv_blocks = int(path.stem.split("_")[1][0])
    assert sum(layer.__class__.__name__ == "Conv2D" for layer in model.layers) == conv_blocks
    assert model.input_shape == (None, 750, 750, 1)
    saved = []
    with h5py.File(str(path), "r") as f:
        weights = f["model_weights"]
        for layer_name in weights.attrs["layer_names"]:
            group = weights[layer_name]
            for weight_name in group.attrs["weight_names"]:
                saved.append(np.asarray(group[weight_name]))
    loaded = model.get_weights()
    assert len(saved) == len(loaded)
    for expected, found in zip(saved, loaded):
        np.testing.assert_array_equal(expected, found)


@pytest.mark.parametrize("path", CNN_FILES, ids=lambda p: p.stem)
def test_2019_cnns_were_trained_in_batches_of_24(path):
    # The optimizer state counts 218 updates for the one epoch over the 5,232 training images.
    # The script committed in 2019 asked for batches of 32, which would give 164.
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "r") as f:
        names = []
        f["optimizer_weights"].visit(names.append)
        iterations = [int(f["optimizer_weights"][n][()]) for n in names if "iterations" in n]
    assert iterations == [5232 // 24]


def test_2019_cnns_label_every_image_pneumonia():
    """The three saved CNNs never learned: their output stays below 0.5 (pneumonia)."""
    keras_or_skip()
    from pneumonia_cnn import load_2019_cnn

    side = np.linspace(0, 1, 750)
    gradient = np.tile(side, (750, 1))
    rng = np.random.default_rng(0)
    lungs = np.clip(
        0.2
        + 0.6 * np.exp(-(((gradient - 0.3) / 0.15) ** 2 + ((gradient.T - 0.5) / 0.3) ** 2))
        + 0.6 * np.exp(-(((gradient - 0.7) / 0.15) ** 2 + ((gradient.T - 0.5) / 0.3) ** 2)),
        0,
        1,
    )
    images = np.stack(
        [np.zeros((750, 750)), np.ones((750, 750)), rng.random((750, 750)), gradient, lungs]
    )
    for path in CNN_FILES:
        normal_probability = load_2019_cnn(str(path)).predict(
            images[..., None], batch_size=1, verbose=0
        )
        assert normal_probability.max() < 0.5, path.stem


def test_benchmark_scores_the_2019_cnns_as_the_pneumonia_share(tmp_path):
    keras_or_skip()
    labels = [0, 0, 1, 0, 1, 0] * 4  # 16 of 24 pneumonia: the always-pneumonia accuracy is 2/3
    write_pipeline_pickle(tmp_path / "train_images.pickle", labels, size=750)
    write_pipeline_pickle(tmp_path / "test_images.pickle", labels, size=750, seed=5)
    run_script(
        REPO / "ML_Models_Saved" / "Model_Benchmark.py",
        "--data-dir",
        tmp_path,
        "--models",
        "*.h5",
        "--out",
        tmp_path / "cnn.xlsx",
        "--metrics-json",
        tmp_path / "cnn.json",
    )
    metrics = json.loads((tmp_path / "cnn.json").read_text())
    assert sorted(metrics) == ["CNN_2layers_1epoch", "CNN_3layers_1epoch", "CNN_4layers_1epoch"]
    for model in metrics.values():
        assert model["avg_partition_accuracy"] == pytest.approx(16 / 24)
        assert model["sensitivity"] == 1.0 and model["specificity"] == 0.0
