"""Synthetic data in the formats the project uses, and a runner for the repository's scripts."""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
HPC = REPO / "HPC"
CNN_MODEL = REPO / "CNN_Model"
SAVED_MODELS = REPO / "ML_Models_Saved"


def synthetic_pixels(rng, label, size):
    """
    Noise; normal (1) images have a brighter center. The difference survives the per-image
    standardization of the "SGD" pipeline, so every model can separate the classes.
    """
    pixels = rng.integers(0, 150, size=(size, size))
    if label == 1:
        low, high = size // 4, size - size // 4
        pixels[low:high, low:high] += 100
    return pixels.astype(np.uint8)


def write_pipeline_pickle(path, labels, size=16, seed=0):
    """A file in the format of PictureProcessor.py: a count, then [label, PIL image] pairs."""
    rng = np.random.default_rng(seed)
    with open(str(path), "wb") as f:
        pickle.dump(len(labels), f)
        for label in labels:
            pickle.dump([int(label), Image.fromarray(synthetic_pixels(rng, label, size))], f)


def mixed_labels(n_pneumonia, n_normal, seed=0):
    labels = np.array([0] * n_pneumonia + [1] * n_normal)
    np.random.default_rng(seed).shuffle(labels)
    return labels.tolist()


def write_jpeg_dataset(root, counts, seed=0):
    """A chest_xray-like folder of small JPEGs, some in color; counts maps "train/NORMAL" to n."""
    rng = np.random.default_rng(seed)
    for folder, n in counts.items():
        label = 1 if folder.endswith("NORMAL") else 0
        (root / folder).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            width, height = int(rng.integers(40, 90)), int(rng.integers(40, 90))
            square = synthetic_pixels(rng, label, max(width, height))
            pixels = np.ascontiguousarray(square[:height, :width])
            image = Image.fromarray(pixels)
            if i % 4 == 3:
                image = image.convert("RGB")
            prefix = "NORMAL" if label else "person"
            image.save(str(root / folder / f"{prefix}{i}-{label}.jpeg"), quality=90)
    return root


def run_script(script, *args, cwd=None, check=True):
    """Runs a script of the repository with HPC/ and CNN_Model/ on the import path."""
    env = dict(os.environ)
    paths = [str(HPC), str(CNN_MODEL)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    command = [sys.executable, str(script)] + [str(a) for a in args]
    result = subprocess.run(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if check and result.returncode != 0:
        raise AssertionError(f"{script} failed:\n{result.stdout}")
    return result
