"""
Project Marrow beta demo: shows one test image, then the Naive Bayes and SVM predictions
for it and each model's accuracy on that image's test partition.

The 2019 SVM loads only in the Python 3.8 environment of requirements-2019.txt:

python Tests_and_Demos/Beta_Presentation_Demo.py --data-dir DIR
"""
from ML_DataPipeline import DataPipeline
from PIL import Image
import _pickle as pickle
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Project Marrow beta demo: Naive Bayes and SVM on one test image.")
parser.add_argument("--data-dir", default=".",
                    help="folder with train_images.pickle and test_images.pickle (default: current folder)")
parser.add_argument("--models-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ML_Models_Saved"),
                    help="folder with NBClassifier_1.pickle and SVMClassifier_1.pickle (default: ML_Models_Saved)")
parser.add_argument("--partition", type=int, help="test partition to use (default: a random one)")
parser.add_argument("--no-show", action="store_true", help="do not open the image or wait for Enter")
args = parser.parse_args()

path = args.data_dir
train_path = os.path.join(path, "train_images.pickle")
test_path = os.path.join(path, "test_images.pickle")
nb_path = os.path.join(args.models_dir, "NBClassifier_1.pickle")

svm_path = os.path.join(args.models_dir, "SVMClassifier_1.pickle")

with open(nb_path, "rb") as model :
    nb_classifier = pickle.load(model)
pipeline = DataPipeline("NB", train_file_path=train_path, test_file_path=test_path)
partitions = pipeline.get_partitions()


random_batch = np.random.randint(0, partitions) if args.partition is None else args.partition
X_, y_ = pipeline.get_test_data(random_batch)

side = int(np.sqrt(X_.shape[1]))
sample_image = X_[0].reshape((side, side))
sample_label = y_[0]

y_hat = nb_classifier.predict([X_[0]])
accuracy = nb_classifier.score(X_, y_)
results = pd.DataFrame({"Prediction": y_hat, "Actual": y_[0]})

print("Project Marrow Beta Demo: Multinomial Naive Bayes")
print("-------------------------------------------------")
if sample_label == 0: print("Sample Image Displays Pneumonia")
else: print("Sample Image Displays Normal")
if not args.no_show:
    im = Image.fromarray(sample_image.astype(np.uint8))
    im.show()
    input("\n")
print("\nNB Prediction on Sample Image")
print(results)
print("Benchmark Prediction on Test Subset: %f" %accuracy)

with open(svm_path, "rb") as model :
    svm_classifier = pickle.load(model)
# The SVM was trained on standardized pixels, so it gets the same partition from the "SGD" pipeline
pipeline = DataPipeline("SGD", train_file_path=train_path, test_file_path=test_path)
X_, y_ = pipeline.get_test_data(random_batch)

y_hat = svm_classifier.predict([X_[0]])
accuracy = svm_classifier.score(X_, y_)
results = pd.DataFrame({"Prediction": y_hat, "Actual": y_[0]})

print("\nSVM Prediction on Sample Image")
print(results)
print("Benchmark Prediction on Test Subset: %f" %accuracy)
