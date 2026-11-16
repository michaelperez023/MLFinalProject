"""
The 2019 training scripts in HPC/, the benchmark and the presentation demo, run as scripts on
synthetic data. The 2019 models themselves are tested in test_saved_models.py.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from helpers import HPC, REPO, run_script


def load_single_pickle(path):
    with open(str(path), "rb") as f:
        model = pickle.load(f)
        assert f.read() == b"", "the file holds more than one object"
    return model


@pytest.mark.parametrize(
    "script,loss",
    [("SVMClassifier.py", "hinge"), ("PERClassifier.py", "perceptron"), ("LOGClassifier.py", None)],
)
def test_sgd_scripts_save_the_epoch_checkpoints(pipeline_dir, tmp_path, script, loss):
    out = tmp_path / "models"
    out.mkdir()
    for _ in range(2):  # a second run must replace the files, not append to them
        run_script(HPC / script, "--data-dir", pipeline_dir, "--out-dir", out, "--epochs", 3)
    prefix = script.replace(".py", "")
    assert sorted(p.name for p in out.iterdir()) == [prefix + "_1.pickle", prefix + "_3.pickle"]
    model = load_single_pickle(out / (prefix + "_3.pickle"))
    from sklearn.linear_model import SGDClassifier

    expected = loss or ("log_loss" if "log_loss" in SGDClassifier.loss_functions else "log")
    assert model.loss == expected
    assert model.random_state == 42
    # 3 epochs of 100 images in 4 batches of 24 to 28
    assert model.t_ == 3 * 100 + 1


def test_naive_bayes_script(pipeline_dir, tmp_path):
    for _ in range(2):
        run_script(HPC / "NBClassifier.py", "--data-dir", pipeline_dir, "--out-dir", tmp_path)
    model = load_single_pickle(tmp_path / "NBClassifier_1.pickle")
    assert model.class_count_.tolist() == [74, 26]


def test_training_scripts_stop_on_an_empty_dataset(pipeline_dir, tmp_path):
    # What the 2019 PictureProcessor.py wrote when it found no images; the 2019 scripts then
    # saved untrained models and exited without an error
    with open(str(pipeline_dir / "train_images.pickle"), "wb") as f:
        pickle.dump(0, f)
    out = tmp_path / "models"
    for script in ("NBClassifier.py", "SVMClassifier.py"):
        result = run_script(HPC / script, "--data-dir", pipeline_dir, "--out-dir", out, check=False)
        assert result.returncode != 0 and "number of images, found 0" in result.stdout
    assert list(out.iterdir()) == []


def train_models(pipeline_dir, out):
    """NB and SVM models with the 2019 file names, trained on the synthetic data."""
    run_script(HPC / "NBClassifier.py", "--data-dir", pipeline_dir, "--out-dir", out)
    run_script(
        HPC / "SVMClassifier.py", "--data-dir", pipeline_dir, "--out-dir", out, "--epochs", 1
    )


def test_benchmark_scores_and_metrics(pipeline_dir, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    train_models(pipeline_dir, models)
    out = tmp_path / "Test_Results.xlsx"
    metrics_path = tmp_path / "metrics.json"
    result = run_script(
        REPO / "ML_Models_Saved" / "Model_Benchmark.py",
        "--data-dir",
        pipeline_dir,
        "--models-dir",
        models,
        "--models",
        "*.pickle",
        "--out",
        out,
        "--metrics-json",
        metrics_path,
    )
    assert "Testing Complete!" in result.stdout

    table = pd.read_excel(str(out), sheet_name="Results", index_col="Test_Batch", engine="openpyxl")
    assert list(table.columns) == ["NBClassifier_1", "SVMClassifier_1"]
    assert list(table.index) == list(range(1, 13)) + ["AVG"]
    np.testing.assert_allclose(table.loc["AVG"], table.iloc[:12].mean())

    metrics = json.loads(metrics_path.read_text())
    for name, model in metrics.items():
        matrix = model["confusion_matrix"]
        tp = matrix["true_pneumonia"]["predicted_pneumonia"]
        fn = matrix["true_pneumonia"]["predicted_normal"]
        fp = matrix["true_normal"]["predicted_pneumonia"]
        tn = matrix["true_normal"]["predicted_normal"]
        assert tp + fn == 31 and fp + tn == 19
        assert model["correct"] == tp + tn
        assert model["accuracy"] == pytest.approx((tp + tn) / 50)
        assert model["sensitivity"] == pytest.approx(tp / 31)
        assert model["specificity"] == pytest.approx(tn / 19)
        assert model["partition_accuracies"] == pytest.approx(table[name].iloc[:12].tolist())
        # The synthetic normal images have a brighter center, which both models pick up
        assert model["roc_auc"] > 0.9


def test_benchmark_without_models_stops(pipeline_dir, tmp_path):
    # The data files in the same folder are not taken for models
    result = run_script(
        REPO / "ML_Models_Saved" / "Model_Benchmark.py",
        "--data-dir",
        pipeline_dir,
        "--models-dir",
        pipeline_dir,
        "--out",
        tmp_path / "x.xlsx",
        check=False,
    )
    assert result.returncode != 0 and "No models match" in result.stdout
    (pipeline_dir / "notes.pickle").write_bytes(pickle.dumps("not a model"))
    result = run_script(
        REPO / "ML_Models_Saved" / "Model_Benchmark.py",
        "--data-dir",
        pipeline_dir,
        "--models-dir",
        pipeline_dir,
        "--out",
        tmp_path / "x.xlsx",
        check=False,
    )
    assert result.returncode != 0 and "does not hold a saved model" in result.stdout


def test_presentation_demo(pipeline_dir, tmp_path):
    train_models(pipeline_dir, tmp_path)
    result = run_script(
        REPO / "Tests_and_Demos" / "Beta_Presentation_Demo.py",
        "--data-dir",
        pipeline_dir,
        "--models-dir",
        tmp_path,
        "--partition",
        11,
        "--no-show",
    )
    assert "NB Prediction on Sample Image" in result.stdout
    assert "SVM Prediction on Sample Image" in result.stdout
    accuracies = [
        float(line.rsplit(":", 1)[1])
        for line in result.stdout.splitlines()
        if line.startswith("Benchmark Prediction")
    ]
    # The SVM is scored on the standardized pixels it was trained on (in 2019 it got raw pixels)
    from ML_DataPipeline import DataPipeline

    svm = load_single_pickle(tmp_path / "SVMClassifier_1.pickle")
    features, labels = DataPipeline(
        "SGD",
        train_file_path=str(pipeline_dir / "train_images.pickle"),
        test_file_path=str(pipeline_dir / "test_images.pickle"),
    ).get_test_data(11)
    assert accuracies[1] == pytest.approx(svm.score(features, labels), abs=1e-6)
