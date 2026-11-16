"""
2026 addition: retrains the 2019 CNN architecture (pneumonia_cnn.build_model) with changes
meant to let it learn, and scores it on the dataset's official test split.

What changed from the 2019 training:

- 256x256 inputs (resized from the original JPEGs with bilinear filtering) instead of 750x750
- a stratified validation split of 10% of the training images; the test split is used once,
  for the final scores
- class weights, because 74% of the training images show pneumonia
- light augmentation during training: small rotations, shifts, zoom and contrast changes
- Adam (learning rate 1e-3, halved when the validation loss stalls) instead of one pass of
  plain SGD, for up to 40 epochs; the weights of the epoch with the best validation ROC AUC
  are kept, and training stops after 8 epochs without improvement

    python CNN_Model/retrain_2026.py --dataset-dir data/chest_xray --cache-dir cache --out-dir runs

Each run (number of convolutional blocks x seed) gets a folder with metrics.json,
test_scores.csv and model.weights.h5. A run whose metrics.json shows the same settings is
skipped, so an interrupted job can be started again.
Pneumonia is the positive class: sensitivity is the share of pneumonia images labelled
pneumonia, specificity the share of normal images labelled normal.
"""

import argparse
import csv
import json
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

CLASSES = {"PNEUMONIA": 0, "NORMAL": 1}
SPLITS = ("train", "test")


def list_images(dataset_dir):
    """Image paths (relative to dataset_dir) and labels of each split, in sorted order."""
    dataset_dir = Path(dataset_dir)
    found = {}
    for split in SPLITS:
        files, labels = [], []
        for folder, label in sorted(CLASSES.items()):
            names = sorted(p.name for p in (dataset_dir / split / folder).glob("*.jpeg"))
            if not names:
                raise SystemExit(f"No .jpeg images in {dataset_dir / split / folder}")
            files += [f"{split}/{folder}/{name}" for name in names]
            labels += [label] * len(names)
        found[split] = (files, np.array(labels, dtype=np.int64))
    return found


def load_image(path, size):
    """Grayscale image resized to size x size with bilinear filtering, as uint8."""
    with Image.open(path) as image:
        resized = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _load_image_job(job):
    return load_image(*job)


def load_dataset(dataset_dir, cache_dir, size, workers=8):
    """
    Resized images and labels of both splits. The first call decodes every JPEG and saves the
    arrays to cache_dir; later calls with the same image list and size read the cache.
    """
    listing = list_images(dataset_dir)
    cache = Path(cache_dir) / f"chest_xray_{size}px.npz"
    if cache.exists():
        with np.load(cache) as saved:
            if all(saved[f"files_{split}"].tolist() == listing[split][0] for split in SPLITS):
                return {
                    split: (saved[f"x_{split}"], saved[f"y_{split}"], listing[split][0])
                    for split in SPLITS
                }
        print(f"{cache} does not match the images in {dataset_dir}; rebuilding it")

    arrays = {}
    for split in SPLITS:
        files, labels = listing[split]
        jobs = [(os.path.join(dataset_dir, name), size) for name in files]
        start = time.time()
        if workers > 1:
            # Fresh worker processes: forking a process that already loaded PyTorch can hang
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                images = list(pool.map(_load_image_job, jobs, chunksize=32))
        else:
            images = [_load_image_job(job) for job in jobs]
        arrays[split] = (np.stack(images), labels, files)
        print(
            f"Resized {len(files)} {split} images to {size}x{size} in {time.time() - start:.0f} s"
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.name + ".tmp")
    with open(temporary, "wb") as f:
        np.savez(
            f,
            **{
                f"{key}_{split}": value
                for split, (x, y, files) in arrays.items()
                for key, value in (("x", x), ("y", y), ("files", np.array(files)))
            },
        )
    os.replace(temporary, cache)
    return arrays


def stratified_split(labels, fraction, seed):
    """Indices of a training and a validation part with the same class shares."""
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        cut = max(1, round(fraction * len(indices)))
        validation.append(indices[:cut])
        train.append(indices[cut:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(validation))


def balanced_class_weights(labels):
    """Weights that give both classes the same total weight: n / (2 * n_class)."""
    counts = np.bincount(labels, minlength=2)
    return {label: len(labels) / (2.0 * count) for label, count in enumerate(counts)}


def binary_metrics(labels, normal_probability, threshold=0.5):
    """Scores with pneumonia (label 0) as the positive class."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    normal_probability = np.asarray(normal_probability, dtype=float)
    predicted = (normal_probability > threshold).astype(int)
    tp = int(np.sum((labels == 0) & (predicted == 0)))
    fn = int(np.sum((labels == 0) & (predicted == 1)))
    fp = int(np.sum((labels == 1) & (predicted == 0)))
    tn = int(np.sum((labels == 1) & (predicted == 1)))
    both = 0 < tp + fn < len(labels)
    return {
        "images": len(labels),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "roc_auc": float(roc_auc_score(labels == 0, 1.0 - normal_probability)) if both else None,
        "confusion_matrix": {
            "true_pneumonia": {"predicted_pneumonia": tp, "predicted_normal": fn},
            "true_normal": {"predicted_pneumonia": fp, "predicted_normal": tn},
        },
    }


def accelerator():
    """Name of the GPU the Keras backend uses, or None."""
    import keras

    backend = keras.backend.backend()
    if backend == "torch":
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    elif backend == "tensorflow":
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return gpus[0].name
    elif backend == "jax":
        import jax

        device = jax.devices()[0]
        if device.platform == "gpu":
            return str(device)
    return None


def versions():
    import keras
    import PIL
    import sklearn

    found = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "scikit-learn": sklearn.__version__,
        "keras": keras.__version__,
        "keras_backend": keras.backend.backend(),
    }
    if found["keras_backend"] == "torch":
        import torch

        found["torch"] = torch.__version__
        found["cuda"] = torch.version.cuda
        found["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    return found


def keep_best_weights(model, monitor="val_auc"):
    """Keras callback that keeps the weights of the epoch with the best validation ROC AUC."""
    import keras

    class KeepBestWeights(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = -np.inf
            self.best_epoch = None
            self.weights = None

        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get(monitor)
            if value is not None and value > self.best:
                self.best, self.best_epoch = float(value), epoch + 1
                self.weights = model.get_weights()

    return KeepBestWeights()


def augmentation(seed):
    import keras
    from keras import layers

    return keras.Sequential(
        [
            layers.RandomRotation(0.03, fill_mode="nearest", seed=seed),  # up to about 11 degrees
            layers.RandomTranslation(0.05, 0.05, fill_mode="nearest", seed=seed + 1),
            layers.RandomZoom(0.1, fill_mode="nearest", seed=seed + 2),
            layers.RandomContrast(0.1, value_range=(0, 1), seed=seed + 3),
        ],
        name="augmentation",
    )


def run_settings(args):
    return {
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.val_fraction,
        "early_stopping_patience": args.patience,
    }


def finished(run_dir, args):
    """True if run_dir holds the metrics of a run with the current settings."""
    path = run_dir / "metrics.json"
    if not path.exists():
        return False
    metrics = json.loads(path.read_text())
    settings = {k: v for k, v in metrics.get("settings", {}).items() if k != "class_weight"}
    return settings == run_settings(args) and metrics.get("image_size") == args.image_size


def train_run(data, conv_blocks, seed, args, run_dir):
    import keras
    from pneumonia_cnn import build_model

    keras.utils.set_random_seed(seed)
    x_all, y_all, _ = data["train"]
    x_test, y_test, test_files = data["test"]
    train_index, val_index = stratified_split(y_all, args.val_fraction, seed)

    def to_input(x):
        return (x.astype(np.float32) / 255.0)[..., None]

    x_train, y_train = to_input(x_all[train_index]), y_all[train_index]
    x_val, y_val = to_input(x_all[val_index]), y_all[val_index]
    class_weight = balanced_class_weights(y_train)

    size = x_all.shape[1]
    model = build_model(conv_blocks, input_shape=(size, size, 1))
    trainer = keras.Sequential([keras.Input((size, size, 1)), augmentation(seed), model])
    trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")],
    )
    best = keep_best_weights(model)
    callbacks = [
        best,
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=args.patience),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    gpu = accelerator()
    torch_cuda = keras.backend.backend() == "torch" and gpu is not None
    if torch_cuda:
        import torch

        torch.cuda.reset_peak_memory_stats()
    peak_gpu_memory_gib = None
    start = time.time()
    history = trainer.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.time() - start
    if torch_cuda:
        peak_gpu_memory_gib = torch.cuda.max_memory_allocated() / 2**30
    if best.weights is not None:
        model.set_weights(best.weights)

    start = time.time()
    p_val = np.asarray(model.predict(x_val, batch_size=64, verbose=0)).reshape(-1)
    p_test = np.asarray(model.predict(to_input(x_test), batch_size=64, verbose=0)).reshape(-1)
    predict_seconds = time.time() - start

    metrics = {
        "conv_blocks": conv_blocks,
        "seed": seed,
        "image_size": size,
        "settings": {
            **run_settings(args),
            "class_weight": {str(k): v for k, v in class_weight.items()},
        },
        "images": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "epochs_run": len(history.history["loss"]),
        "best_epoch": best.best_epoch,
        "parameters": int(model.count_params()),
        "validation": binary_metrics(y_val, p_val),
        "test": binary_metrics(y_test, p_test),
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "device": gpu or "cpu",
        "peak_gpu_memory_gib": peak_gpu_memory_gib,
        "versions": versions(),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(run_dir / "model.weights.h5")
    with open(run_dir / "test_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "normal_probability"])
        writer.writerows(zip(test_files, y_test.tolist(), p_test.tolist(), strict=True))
    temporary = run_dir / "metrics.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=1) + "\n")
    os.replace(temporary, run_dir / "metrics.json")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Retrain the 2019 CNN architecture with a validation split, class weights, "
        "augmentation and Adam, and score it on the test split."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="the dataset's chest_xray folder (train/ and test/, each with NORMAL/ and PNEUMONIA/)",
    )
    parser.add_argument("--cache-dir", required=True, help="folder for the resized images")
    parser.add_argument("--out-dir", required=True, help="folder for the runs")
    parser.add_argument(
        "--conv-blocks",
        type=int,
        nargs="+",
        choices=(2, 3, 4),
        default=[2, 3, 4],
        help="CNN depths to train (default: 2 3 4)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="seeds (default: 42)")
    parser.add_argument(
        "--image-size", type=int, default=256, help="input size in pixels (default: 256)"
    )
    parser.add_argument("--epochs", type=int, default=40, help="maximum epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size (default: 32)")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Adam learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="share of the training images used for validation (default: 0.1)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="epochs without a better validation ROC AUC before stopping (default: 8)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="processes for resizing the images (default: up to 8)",
    )
    parser.add_argument("--require-gpu", action="store_true", help="stop if Keras cannot use a GPU")
    args = parser.parse_args(argv)

    gpu = accelerator()
    print(f"Keras backend {versions()['keras_backend']}, device: {gpu or 'CPU'}")
    if args.require_gpu and gpu is None:
        raise SystemExit(
            "No GPU available to Keras (run without --require-gpu to train on the CPU)"
        )

    data = load_dataset(args.dataset_dir, args.cache_dir, args.image_size, args.workers)
    out_dir = Path(args.out_dir)
    for conv_blocks in args.conv_blocks:
        for seed in args.seeds:
            run_dir = out_dir / f"cnn{conv_blocks}_seed{seed}"
            if finished(run_dir, args):
                print(f"{run_dir} is done; skipping")
                continue
            print(f"Training {conv_blocks} convolutional blocks, seed {seed}")
            metrics = train_run(data, conv_blocks, seed, args, run_dir)
            scores = ", ".join(
                f"{name} {metrics['test'][name]:.4f}"
                for name in ("accuracy", "sensitivity", "specificity", "roc_auc")
                if metrics["test"][name] is not None
            )
            print(
                f"{run_dir.name}: test {scores} (best epoch {metrics['best_epoch']} of "
                f"{metrics['epochs_run']}, {metrics['train_seconds']:.0f} s)"
            )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
