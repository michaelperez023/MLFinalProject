"""The CNN script (2019 procedure) and the 2026 retrain, on tiny synthetic data."""

import json

import numpy as np
import pytest
from helpers import mixed_labels, write_pipeline_pickle

keras = pytest.importorskip("keras")

import pneumonia_cnn  # noqa: E402
import retrain_2026  # noqa: E402

# Parameters of the 2019 models, from their saved weights
PARAMETERS_2019 = {2: 7233, 3: 27777, 4: 105729}


@pytest.mark.parametrize("conv_blocks", [2, 3, 4])
def test_build_model(conv_blocks):
    model = pneumonia_cnn.build_model(conv_blocks)
    assert model.input_shape == (None, 750, 750, 1)
    assert model.output_shape == (None, 1)
    assert model.count_params() == PARAMETERS_2019[conv_blocks]
    # Global average pooling makes the architecture independent of the input size
    assert (
        pneumonia_cnn.build_model(conv_blocks, (64, 64, 1)).count_params()
        == PARAMETERS_2019[conv_blocks]
    )


def test_build_model_rejects_other_depths():
    with pytest.raises(ValueError, match="2, 3 or 4"):
        pneumonia_cnn.build_model(5)


def test_load_2019_cnn_rejects_other_files(tmp_path):
    model = keras.Sequential([keras.Input((8,)), keras.layers.Dense(1)])
    path = tmp_path / "other.h5"
    model.save(str(path))
    with pytest.raises(ValueError, match="does not hold a 2019 Project Marrow CNN"):
        pneumonia_cnn.load_2019_cnn(str(path))


def test_2019_training_procedure(tmp_path):
    write_pipeline_pickle(tmp_path / "train_images.pickle", mixed_labels(20, 12), size=24)
    write_pipeline_pickle(tmp_path / "test_images.pickle", mixed_labels(8, 4, seed=1), size=24)
    out = tmp_path / "cnn.keras"
    pneumonia_cnn.main(
        ["--data-dir", str(tmp_path), "--conv-blocks", "3", "--batch-size", "8", "--out", str(out)]
    )
    model = keras.models.load_model(str(out))
    assert model.input_shape == (None, 24, 24, 1)
    assert model.count_params() == PARAMETERS_2019[3]


def test_default_batch_size_is_that_of_the_2019_models(tmp_path):
    # One plain SGD update per 24 images, as in the optimizer state of the saved 2019 CNNs
    # (tests/test_saved_models.py)
    write_pipeline_pickle(tmp_path / "train_images.pickle", mixed_labels(30, 18), size=16)
    write_pipeline_pickle(tmp_path / "test_images.pickle", mixed_labels(4, 4, seed=1), size=16)
    model = pneumonia_cnn.main(["--data-dir", str(tmp_path), "--out", str(tmp_path / "cnn.keras")])
    assert int(keras.ops.convert_to_numpy(model.optimizer.iterations)) == 48 // 24


# 2026 retrain ---------------------------------------------------------------------------------


def test_stratified_split_keeps_class_shares():
    labels = np.array([0] * 300 + [1] * 100)
    train, validation = retrain_2026.stratified_split(labels, 0.1, seed=3)
    assert len(set(train) & set(validation)) == 0
    assert len(train) + len(validation) == 400
    assert np.bincount(labels[validation]).tolist() == [30, 10]
    again = retrain_2026.stratified_split(labels, 0.1, seed=3)
    assert np.array_equal(again[1], validation)


def test_balanced_class_weights():
    weights = retrain_2026.balanced_class_weights(np.array([0] * 3883 + [1] * 1349))
    assert weights[0] == pytest.approx(5232 / (2 * 3883))
    assert weights[1] == pytest.approx(5232 / (2 * 1349))
    assert weights[0] * 3883 == pytest.approx(weights[1] * 1349)


def test_binary_metrics_with_pneumonia_as_positive_class():
    labels = np.array([0, 0, 0, 1, 1])
    # p(normal) for each image: one pneumonia image missed, one normal image flagged
    normal_probability = np.array([0.1, 0.2, 0.8, 0.3, 0.9])
    metrics = retrain_2026.binary_metrics(labels, normal_probability)
    assert metrics["confusion_matrix"] == {
        "true_pneumonia": {"predicted_pneumonia": 2, "predicted_normal": 1},
        "true_normal": {"predicted_pneumonia": 1, "predicted_normal": 1},
    }
    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert metrics["sensitivity"] == pytest.approx(2 / 3)
    assert metrics["specificity"] == pytest.approx(1 / 2)
    assert metrics["precision"] == pytest.approx(2 / 3)
    # Pairs (pneumonia, normal) ranked correctly: 5 of 6
    assert metrics["roc_auc"] == pytest.approx(5 / 6)


def test_always_pneumonia_scores_the_pneumonia_share():
    metrics = retrain_2026.binary_metrics(np.array([0] * 390 + [1] * 234), np.full(624, 0.3))
    assert metrics["accuracy"] == pytest.approx(0.625)
    assert metrics["sensitivity"] == 1.0 and metrics["specificity"] == 0.0
    assert metrics["roc_auc"] == pytest.approx(0.5)


def test_load_dataset_builds_and_reuses_the_cache(jpeg_dataset, tmp_path):
    data = retrain_2026.load_dataset(jpeg_dataset, tmp_path / "cache", 20, workers=1)
    x_train, y_train, files = data["train"]
    assert x_train.shape == (42, 20, 20) and x_train.dtype == np.uint8
    assert np.bincount(y_train).tolist() == [30, 12]
    assert files[0].startswith("train/NORMAL/")
    cache = tmp_path / "cache" / "chest_xray_20px.npz"
    stamp = cache.stat().st_mtime_ns
    again = retrain_2026.load_dataset(jpeg_dataset, tmp_path / "cache", 20, workers=1)
    assert cache.stat().st_mtime_ns == stamp
    assert np.array_equal(again["test"][0], data["test"][0])


def test_retrain_runs_and_skips_finished_runs(jpeg_dataset, tmp_path, capsys):
    args = [
        "--dataset-dir",
        str(jpeg_dataset),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out-dir",
        str(tmp_path / "runs"),
        "--conv-blocks",
        "2",
        "--seeds",
        "1",
        "--image-size",
        "24",
        "--epochs",
        "2",
        "--batch-size",
        "8",
        "--workers",
        "1",
    ]
    retrain_2026.main(args)
    run = tmp_path / "runs" / "cnn2_seed1"
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["images"] == {"train": 38, "validation": 4, "test": 18}
    assert metrics["epochs_run"] <= 2 and metrics["best_epoch"] in (1, 2)
    test = metrics["test"]
    matrix = test["confusion_matrix"]
    assert sum(matrix["true_pneumonia"].values()) == 10
    assert sum(matrix["true_normal"].values()) == 8
    assert 0 <= test["roc_auc"] <= 1
    assert (run / "model.weights.h5").exists()
    assert len((run / "test_scores.csv").read_text().splitlines()) == 19

    capsys.readouterr()
    retrain_2026.main(args)
    assert "is done; skipping" in capsys.readouterr().out

    # Other settings: the run is trained again
    retrain_2026.main(args[:-6] + ["--epochs", "1", "--batch-size", "8", "--workers", "1"])
    assert json.loads((run / "metrics.json").read_text())["settings"]["max_epochs"] == 1
