"""Data_Preprocessing/PictureProcessor.py on a small folder of synthetic JPEGs."""

import importlib.util
import pickle

import numpy as np
import pytest
from helpers import REPO
from PIL import Image


def load_processor():
    spec = importlib.util.spec_from_file_location(
        "PictureProcessor", str(REPO / "Data_Preprocessing" / "PictureProcessor.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_pickle(path):
    with open(str(path), "rb") as f:
        count = pickle.load(f)
        items = [pickle.load(f) for _ in range(count)]
        trailing = f.read()
    return count, items, trailing


def test_writes_the_pipeline_format(jpeg_dataset, tmp_path):
    out = tmp_path / "out"
    load_processor().main(
        ["--dataset-dir", str(jpeg_dataset), "--out-dir", str(out), "--size", "24", "--seed", "1"]
    )
    count, items, trailing = read_pickle(out / "train_images.pickle")
    assert count == 42 and len(items) == 42 and trailing == b""
    assert sorted(label for label, _ in items) == [0] * 30 + [1] * 12
    assert all(image.mode == "L" and image.size == (24, 24) for _, image in items)
    count, items, trailing = read_pickle(out / "test_images.pickle")
    assert count == 18 and sorted(label for label, _ in items) == [0] * 10 + [1] * 8
    assert not list(out.glob("*.tmp"))


def test_resizes_with_nearest_neighbour(tmp_path):
    folders = {}
    for split in ("train", "test"):
        for name in ("NORMAL", "PNEUMONIA"):
            folders[split, name] = tmp_path / "chest_xray" / split / name
            folders[split, name].mkdir(parents=True)
    # JPEG is lossy, so the expected pixels come from decoding the same file
    pixels = np.tile(np.arange(0, 250, 25, dtype=np.uint8), (10, 1))
    source = folders["train", "NORMAL"] / "a.jpeg"
    Image.fromarray(pixels).save(str(source), quality=100)
    for key, folder in folders.items():
        if key != ("train", "NORMAL"):
            Image.fromarray(pixels).save(str(folder / "b.jpeg"), quality=100)
    load_processor().main(
        ["--dataset-dir", str(tmp_path / "chest_xray"), "--out-dir", str(tmp_path), "--size", "7"]
    )
    with Image.open(str(source)) as image:
        expected = np.asarray(
            image.convert("L").resize((7, 7), getattr(Image, "Resampling", Image).NEAREST)
        )
    _, items, _ = read_pickle(tmp_path / "train_images.pickle")
    normal = [np.asarray(image) for label, image in items if label == 1]
    np.testing.assert_array_equal(normal[0], expected)


def test_seeded_runs_are_identical_and_reruns_overwrite(jpeg_dataset, tmp_path):
    processor = load_processor()
    args = ["--dataset-dir", str(jpeg_dataset), "--size", "16", "--seed", "7"]
    processor.main(args + ["--out-dir", str(tmp_path / "a")])
    processor.main(args + ["--out-dir", str(tmp_path / "b")])
    first = (tmp_path / "a" / "train_images.pickle").read_bytes()
    assert first == (tmp_path / "b" / "train_images.pickle").read_bytes()
    # The 2019 version appended to existing files, so a second run doubled them
    processor.main(args + ["--out-dir", str(tmp_path / "a")])
    assert (tmp_path / "a" / "train_images.pickle").read_bytes() == first


def test_missing_images_stop_with_a_clear_message(tmp_path):
    (tmp_path / "chest_xray" / "train" / "NORMAL").mkdir(parents=True)
    with pytest.raises(SystemExit, match="No images match"):
        load_processor().main(
            ["--dataset-dir", str(tmp_path / "chest_xray"), "--out-dir", str(tmp_path)]
        )
    assert not (tmp_path / "train_images.pickle").exists()
