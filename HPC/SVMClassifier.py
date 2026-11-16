"""
@Author : Peter Akioyamen
@Code : CAP 4612

This python script is a part of the model development process.
It acts to instantiate and train 3
classification models which use Gradient Descent as the
learning algorithm, enabling mini-batch training.
"""

from ML_DataPipeline import DataPipeline
from sklearn.linear_model import SGDClassifier
import _pickle as pickle
import argparse
import os

parser = argparse.ArgumentParser(description="Train the linear SVM (SGDClassifier with hinge loss)"
                                             " for --epochs passes over the training data.")
parser.add_argument("--data-dir", default=".", help="folder with train_images.pickle (default: current folder)")
parser.add_argument("--out-dir", default=".", help="folder for the saved models (default: current folder)")
parser.add_argument("--epochs", type=int, default=200, help="passes over the training data (default: 200)")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

labels = [0, 1]
epochs = args.epochs
seed = 42

# The 2019 runs asked for batch_size=96, which DataPipeline replaces with its default of 24
# (see the warning in Training_Logs), so the saved models were trained with batches of 24
pipeline = DataPipeline("SGD", batch_size=24, train_file_path=os.path.join(args.data_dir, "train_images.pickle"),
                        test_file_path=os.path.join(args.data_dir, "test_images.pickle"))

print("Support Vector Machine")
print("----------------------")
svm = SGDClassifier(verbose=0, random_state=seed)
class_flag = True
for epoch in range(epochs):
    if epoch % 20 == 0:
        print("Epoch: %d" %(epoch + 1))
    if epoch == (epochs - 1):
        print("Epoch: %d" %epochs)

    iterations = pipeline.get_iterations()
    for iters in range(iterations):
        x_train, y_train = pipeline.get_training_batch(iters)
        if class_flag:
            svm.partial_fit(x_train, y_train, classes=labels)
            class_flag = False
            continue
        svm.partial_fit(x_train, y_train)

    if epoch == 0:
        with open(os.path.join(args.out_dir, "SVMClassifier_" + str(epoch + 1) + ".pickle"), "wb") as model_file:
            pickle.dump(svm, model_file)
    if epoch == 49:
        with open(os.path.join(args.out_dir, "SVMClassifier_" + str(epoch + 1) + ".pickle"), "wb") as model_file:
            pickle.dump(svm, model_file)
    if epoch == 99:
        with open(os.path.join(args.out_dir, "SVMClassifier_" + str(epoch + 1) + ".pickle"), "wb") as model_file:
            pickle.dump(svm, model_file)
    if epoch == (epochs - 1):
        with open(os.path.join(args.out_dir, "SVMClassifier_" + str(epochs) + ".pickle"), "wb") as model_file:
            pickle.dump(svm, model_file)
