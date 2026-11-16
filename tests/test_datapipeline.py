"""
DataPipeline on synthetic data. The first tests are the checks of Peter Akioyamen's
Tests_and_Demos/Test_DataPipeline_Class.py (2019), which could no longer run because it
created DataPipeline() without its required model argument.
"""

import importlib
import pickle
import warnings

import numpy as np
import pytest
from helpers import write_pipeline_pickle
from ML_DataPipeline import DataPipeline

pipeline_module = importlib.import_module("ML_DataPipeline.DataPipeline")


def make_pipeline(directory, model="SGD", **kwargs):
    return DataPipeline(
        model,
        train_file_path=str(directory / "train_images.pickle"),
        test_file_path=str(directory / "test_images.pickle"),
        **kwargs,
    )


# Checks of the 2019 test script ------------------------------------------------------------


def test_training_batches_cover_an_epoch(pipeline_dir):
    pipeline = make_pipeline(pipeline_dir)
    assert pipeline.get_iterations() == 100 // 24
    sizes = []
    for iteration in range(pipeline.get_iterations()):
        features, labels = pipeline.get_training_batch(iteration)
        assert len(features) == len(labels)
        sizes.append(len(labels))
    # The last batch of an epoch also takes the remaining 4 images
    assert sizes == [24, 24, 24, 28]


def test_missing_training_file_raises_oserror(pipeline_dir):
    with pytest.raises(OSError, match="PictureProcessor.py"):
        DataPipeline("SGD", train_file_path=str(pipeline_dir / "DNE"))


def test_test_partitions_cover_the_test_set(pipeline_dir):
    pipeline = make_pipeline(pipeline_dir)
    assert pipeline.get_partitions() == 12
    sizes = []
    for partition in range(pipeline.get_partitions()):
        features, labels = pipeline.get_test_data(partition)
        assert len(features) == len(labels)
        sizes.append(len(labels))
    assert sizes == [4] * 11 + [6]


def test_missing_test_file_raises_oserror(pipeline_dir):
    pipeline = DataPipeline(
        "SGD",
        train_file_path=str(pipeline_dir / "train_images.pickle"),
        test_file_path=str(pipeline_dir / "NULL"),
    )
    with pytest.raises(OSError):
        pipeline.get_test_data(0)


def test_batch_size_checks(pipeline_dir):
    with pytest.raises(ValueError):
        make_pipeline(pipeline_dir, batch_size=32 + 0.5)
    for batch_size in (0, 10000):
        with pytest.warns(UserWarning, match="default will be set"):
            assert make_pipeline(pipeline_dir, batch_size=batch_size).batch_size == 24


def test_partition_checks(pipeline_dir):
    with pytest.raises(ValueError):
        make_pipeline(pipeline_dir, partitions=10 + 0.5)
    with pytest.warns(UserWarning, match="8 or fewer"):
        assert make_pipeline(pipeline_dir, partitions=0).get_partitions() == 12
    with pytest.warns(UserWarning, match="slow down"):
        assert make_pipeline(pipeline_dir, partitions=10000).get_partitions() == 10000


# Features for each model ---------------------------------------------------------------------


def read_all(path):
    with open(str(path), "rb") as f:
        count = pickle.load(f)
        items = [pickle.load(f) for _ in range(count)]
    return np.stack([np.asarray(image) for _, image in items]), np.array(
        [label for label, _ in items]
    )


def test_test_data_keeps_file_order_and_pixels(pipeline_dir):
    images, labels = read_all(pipeline_dir / "test_images.pickle")
    pipeline = make_pipeline(pipeline_dir, model="NB")
    features = np.concatenate([pipeline.get_test_data(p)[0] for p in range(12)])
    targets = np.concatenate([pipeline.get_test_data(p)[1] for p in range(12)])
    np.testing.assert_array_equal(targets, labels)
    np.testing.assert_array_equal(features, images.reshape(len(images), -1))
    assert features.dtype == np.float64


def test_training_batches_hold_each_image_once_per_epoch(pipeline_dir):
    images, labels = read_all(pipeline_dir / "train_images.pickle")
    pipeline = make_pipeline(pipeline_dir, model="NB")
    batches = [pipeline.get_training_batch(i) for i in range(pipeline.get_iterations())]
    features = np.concatenate([b[0] for b in batches])
    targets = np.concatenate([b[1] for b in batches])
    # Batches are shuffled inside, so compare the sets of (pixels, label) rows
    expected = sorted(map(tuple, np.column_stack([images.reshape(len(images), -1), labels])))
    found = sorted(map(tuple, np.column_stack([features, targets]).astype(int)))
    assert found == expected


def test_feature_scaling_per_model(pipeline_dir):
    cnn, _ = make_pipeline(pipeline_dir, model="CNN").get_test_data(0)
    assert cnn.shape == (4, 16, 16, 1)
    assert 0.0 <= cnn.min() and cnn.max() <= 1.0

    sgd, _ = make_pipeline(pipeline_dir, model="SGD").get_test_data(0)
    assert sgd.shape == (4, 256)
    np.testing.assert_allclose(sgd.mean(axis=1), 0, atol=1e-9)
    np.testing.assert_allclose(sgd.std(axis=1), 1, atol=1e-9)

    nb, _ = make_pipeline(pipeline_dir, model="NB").get_test_data(0)
    assert nb.shape == (4, 256)
    assert np.all(nb == np.round(nb)) and nb.min() >= 0 and nb.max() <= 255


def test_out_of_range_requests_name_the_valid_range(pipeline_dir):
    pipeline = make_pipeline(pipeline_dir)
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        pipeline.get_training_batch(4)
    with pytest.raises(ValueError, match=r"\[0, 11\]"):
        pipeline.get_test_data(12)


def test_model_name_is_checked(pipeline_dir):
    with pytest.raises(ValueError, match="NB, SGD, or CNN"):
        make_pipeline(pipeline_dir, model="SVM")


# Reading each file once ----------------------------------------------------------------------


def test_each_pickle_file_is_read_once(pipeline_dir, monkeypatch):
    reads = []
    original = pipeline_module.load_images

    def counting_load(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(pipeline_module, "load_images", counting_load)
    pipeline = make_pipeline(pipeline_dir)
    for _epoch in range(3):
        for iteration in range(pipeline.get_iterations()):
            pipeline.get_training_batch(iteration)
    for partition in range(pipeline.get_partitions()):
        pipeline.get_test_data(partition)
    assert sorted(reads) == sorted([pipeline.train_path, pipeline.test_path])


def test_file_written_twice_is_rejected(pipeline_dir):
    # The 2019 PictureProcessor.py appended to existing files
    first = pipeline_dir / "first.pickle"
    write_pipeline_pickle(first, [0, 1, 0, 1])
    with open(str(pipeline_dir / "train_images.pickle"), "ab") as f:
        f.write(first.read_bytes())
    pipeline = make_pipeline(pipeline_dir)
    with pytest.raises(ValueError, match="written twice"):
        pipeline.get_training_batch(0)


def test_incomplete_file_is_rejected(pipeline_dir):
    path = pipeline_dir / "test_images.pickle"
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="incomplete"):
        make_pipeline(pipeline_dir).get_test_data(0)


def test_empty_dataset_is_rejected(pipeline_dir):
    # What the 2019 PictureProcessor.py wrote when it found no images
    with open(str(pipeline_dir / "test_images.pickle"), "wb") as f:
        pickle.dump(0, f)
    with pytest.raises(ValueError, match="number of images"):
        make_pipeline(pipeline_dir).get_test_data(0)


def test_empty_training_file_is_rejected_at_once(pipeline_dir):
    # Without this check the training loops ran 0 times and the scripts saved untrained models
    path = pipeline_dir / "train_images.pickle"
    with open(str(path), "wb") as f:
        pickle.dump(0, f)
    with pytest.raises(ValueError, match="number of images, found 0"):
        make_pipeline(pipeline_dir)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="is empty"):
        make_pipeline(pipeline_dir)


def test_training_file_smaller_than_one_batch_is_rejected(pipeline_dir):
    write_pipeline_pickle(pipeline_dir / "train_images.pickle", [0, 1] * 5)
    with pytest.raises(ValueError, match="10 images, fewer than one batch of 24"):
        make_pipeline(pipeline_dir)
    assert make_pipeline(pipeline_dir, batch_size=10).get_iterations() == 1


def test_no_warning_for_default_arguments(pipeline_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_pipeline(pipeline_dir)
