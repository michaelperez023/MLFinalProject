"""
@Author : Jeffrey Krakowiak
@Author : Peter Akioyamen
@Code : CAP 4612

This python script is a part of the model development process.
It acts to instantiate and train 4 Multinomial Naive Bayes
classification models which use mini-batch training.
"""

from sklearn.naive_bayes import MultinomialNB
from ML_DataPipeline import DataPipeline
import _pickle as pickle
import argparse
import os

parser = argparse.ArgumentParser(description="Train the multinomial Naive Bayes model with one pass over the data.")
parser.add_argument("--data-dir", default=".", help="folder with train_images.pickle (default: current folder)")
parser.add_argument("--out-dir", default=".", help="folder for NBClassifier_1.pickle (default: current folder)")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

class_flag = True
labels = [0, 1]

pipeline = DataPipeline("NB", batch_size=24, train_file_path=os.path.join(args.data_dir, "train_images.pickle"),
                        test_file_path=os.path.join(args.data_dir, "test_images.pickle"))
nb = MultinomialNB()

print("Multinomial Naive Bayes")
print("-----------------------")
iterations = pipeline.get_iterations()
for iters in range(iterations):
    x_train, y_train = pipeline.get_training_batch(iters)
    if class_flag:
        nb.partial_fit(x_train, y_train, classes=labels)
        class_flag = False
        continue
    nb.partial_fit(x_train, y_train)

with open(os.path.join(args.out_dir, "NBClassifier_1.pickle"), "wb") as model_file:
    pickle.dump(nb, model_file)
