"""
Fixtures with synthetic data. Nothing here downloads the dataset.

The tests of the 2019 code (DataPipeline, PictureProcessor, the training and benchmark
scripts) also run on Python 3.8 in the environment of requirements-2019.txt.
"""

import atexit
import os
import shutil
import tempfile

import pytest
from helpers import mixed_labels, write_jpeg_dataset, write_pipeline_pickle

# Set before any test module imports Keras or Matplotlib, so that they do not write
# configuration files to the home folder
for _name in ("KERAS_HOME", "MPLCONFIGDIR"):
    if _name not in os.environ:
        os.environ[_name] = tempfile.mkdtemp(prefix=_name.lower() + "-")
        atexit.register(shutil.rmtree, os.environ[_name], True)
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def pipeline_dir(tmp_path):
    """train_images.pickle with 100 images and test_images.pickle with 50, at 16x16 pixels."""
    write_pipeline_pickle(tmp_path / "train_images.pickle", mixed_labels(74, 26, seed=1), seed=2)
    write_pipeline_pickle(tmp_path / "test_images.pickle", mixed_labels(31, 19, seed=3), seed=4)
    return tmp_path


@pytest.fixture
def jpeg_dataset(tmp_path):
    counts = {"train/NORMAL": 12, "train/PNEUMONIA": 30, "test/NORMAL": 8, "test/PNEUMONIA": 10}
    return write_jpeg_dataset(tmp_path / "chest_xray", counts)
