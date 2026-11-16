"""
@Author : Peter Akioyamen (2019); changed in 2026, see README
@Code : CAP 4612

Scores every saved model on the 12 partitions of the test set and writes the accuracies,
and their average (AVG), to a spreadsheet. Results/Test_Results.xlsx is the 2019 output.

The 12 SGD models were pickled with scikit-learn 0.21 and load only in the Python 3.8
environment of requirements-2019.txt (the Naive Bayes model also loads in current
scikit-learn); the CNNs (.h5) need Keras 3:

python ML_Models_Saved/Model_Benchmark.py --data-dir DIR --models "*.pickle" --out sklearn.xlsx
python ML_Models_Saved/Model_Benchmark.py --data-dir DIR --models "*.h5" --out cnn.xlsx
"""

from ML_DataPipeline import DataPipeline
import _pickle as pickle
import pandas as pd
import numpy as np
import argparse
import glob
import json
import os


parser = argparse.ArgumentParser(description="Score the saved models on the test set.")
parser.add_argument("--data-dir", default=".",
                    help="folder with train_images.pickle and test_images.pickle (default: current folder)")
parser.add_argument("--models-dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="folder with the saved models (default: the folder of this script)")
parser.add_argument("--models", nargs="+", default=["*.pickle", "*.h5"],
                    help="file name patterns of the models to score (default: *.pickle *.h5)")
parser.add_argument("--out", default="Test_Results.xlsx", help="spreadsheet to write (default: Test_Results.xlsx)")
parser.add_argument("--metrics-json",
                    help="also write each model's confusion matrix, sensitivity, specificity and ROC AUC "
                         "to this JSON file (pneumonia is the positive class)")
args = parser.parse_args()

train_path = os.path.join(args.data_dir, "train_images.pickle")
test_path = os.path.join(args.data_dir, "test_images.pickle")
data_files = ("train_images.pickle", "test_images.pickle")
model_paths = [path for pattern in args.models for path in sorted(glob.glob(os.path.join(args.models_dir, pattern)))
               if os.path.basename(path) not in data_files]
if not model_paths:
    raise SystemExit("No models match " + " ".join(args.models) + " in " + args.models_dir)

models = dict()
accuracies = dict()
for path in model_paths:
    model_name = os.path.splitext(os.path.basename(path))[0]
    if path.endswith(".pickle"):
        with open(path, "rb") as model_file:
            model = pickle.load(model_file)
    elif path.endswith(".h5"):
        # keras.models.load_model() cannot read these Keras 2.3.1 files in Keras 3,
        # so the CNN is rebuilt and the saved weights are loaded into it
        from pneumonia_cnn import load_2019_cnn
        model = load_2019_cnn(path)
    else:
        raise SystemExit("Unknown model file type: " + path)
    if not hasattr(model, "predict"):
        raise SystemExit(path + " does not hold a saved model")
    models[model_name] = model
    accuracies[model_name] = list()


def pipeline_kind(model_name):
    if "CNN" in model_name:
        return "CNN"
    if "NB" in model_name:
        return "NB"
    return "SGD"


def predict(model_name, model, features):
    """Returns the predicted labels and a score that is higher for pneumonia (label 0)."""
    if pipeline_kind(model_name) == "CNN":
        normal_probability = np.asarray(model.predict(features, batch_size=8, verbose=0)).reshape(-1)
        return (normal_probability > 0.5).astype(int), 1.0 - normal_probability
    if hasattr(model, "decision_function"):
        return model.predict(features), -model.decision_function(features)
    log_probability = model.predict_log_proba(features)
    return model.predict(features), log_probability[:, 0] - log_probability[:, 1]


# Only the pipelines that the selected models need are created
pipelines = dict()
for model_name in models:
    kind = pipeline_kind(model_name)
    if kind not in pipelines:
        pipelines[kind] = DataPipeline(kind, batch_size=24, partitions=12, train_file_path=train_path,
                                       test_file_path=test_path)
partitions = 12
labels = {model_name: list() for model_name in models}
predictions = {model_name: list() for model_name in models}
scores = {model_name: list() for model_name in models}

for partition in range(partitions):
    test_data = {kind: pipeline.get_test_data(partition) for kind, pipeline in pipelines.items()}
    print("Partition: %d" %partition)

    for model_name, model in models.items():
        X_test, y_test = test_data[pipeline_kind(model_name)]
        y_pred, y_score = predict(model_name, model, X_test)
        # Accuracy as in 2019: model.score() for scikit-learn, the binary accuracy at 0.5 for the CNNs
        accuracies[model_name].append(float(np.mean(y_pred == y_test)))
        labels[model_name].append(y_test)
        predictions[model_name].append(y_pred)
        scores[model_name].append(y_score)


for model_name in models.keys():
    avg = sum(accuracies[model_name])/len(accuracies[model_name])
    accuracies[model_name].append(avg)

results = pd.DataFrame(accuracies, index=(list(range(1, partitions+1)) + ["AVG"]))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
results.to_excel(args.out, sheet_name="Results", index_label="Test_Batch")

if args.metrics_json:
    from sklearn.metrics import roc_auc_score
    metrics = dict()
    for model_name in models:
        y_true = np.concatenate(labels[model_name])
        y_pred = np.concatenate(predictions[model_name])
        y_score = np.concatenate(scores[model_name])
        # Pneumonia (label 0) is the positive class
        tp = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 0) & (y_pred == 1)))
        fp = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 1) & (y_pred == 1)))
        both_classes = 0 < tp + fn < len(y_true)
        metrics[model_name] = {
            "test_images": int(len(y_true)),
            "correct": tp + tn,
            "accuracy": (tp + tn) / len(y_true),
            "avg_partition_accuracy": accuracies[model_name][-1],
            "partition_accuracies": accuracies[model_name][:-1],
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "roc_auc": float(roc_auc_score(y_true == 0, y_score)) if both_classes else None,
            "confusion_matrix": {"true_pneumonia": {"predicted_pneumonia": tp, "predicted_normal": fn},
                                 "true_normal": {"predicted_pneumonia": fp, "predicted_normal": tn}},
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_json)), exist_ok=True)
    with open(args.metrics_json, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=1)

print("Testing Complete!")
print("-----------------")
