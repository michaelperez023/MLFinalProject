"""
@Author : Peter Akioyamen
@Code : CAP 4612

This python script is a part of the model development process.
It acts to instantiate and train 3
classification models which use Gradient Descent as the
learning algorithm, enabling mini-batch training.
"""

from sklearn.linear_model import SGDClassifier
from ML_DataPipeline import DataPipeline
import _pickle as pickle
import joblib
import os


path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow"
train_path = os.path.join(path, "train_images.pickle")
test_path = os.path.join(path, "test_images.pickle")
class_flag = True
labels = [0, 1]
seed = 42
cpu = joblib.cpu_count()

pipeline = DataPipeline("SGD", batch_size=24, train_file_path=train_path, test_file_path=test_path)
svm = SGDClassifier(verbose=1, n_jobs=(cpu - 2), random_state=seed)
lg = SGDClassifier(loss="log", verbose=1, n_jobs=(cpu - 2), random_state=seed)
per = SGDClassifier(loss="perceptron", verbose=1, n_jobs=(cpu - 2), random_state=seed)

iterations = pipeline.get_iterations()

print("Support Vector Machine")
for iters in range(iterations):
    x_train, y_train = pipeline.get_training_batch(iters)
    if class_flag:
        svm.partial_fit(x_train, y_train, classes=labels)
        class_flag = False
        continue
    svm.partial_fit(x_train, y_train)

class_flag = True
print("Logistic Regression")
for iters in range(iterations):
    x_train, y_train = pipeline.get_training_batch(iters)
    if class_flag:
        lg.partial_fit(x_train, y_train, classes=labels)
        class_flag = False
        continue
    lg.partial_fit(x_train, y_train)

class_flag = True
print("Perceptron")
for iters in range(iterations):
    x_train, y_train = pipeline.get_training_batch(iters)
    if class_flag:
        per.partial_fit(x_train, y_train, classes=labels)
        class_flag = False
        continue
    per.partial_fit(x_train, y_train)

with open("SVMClassifier.pickle", "ab") as model_file:
    pickle.dump(svm, model_file)

with open("LOGClassifier.pickle", "ab") as model_file:
    pickle.dump(lg, model_file)

with open("PERClassifier.pickle", "ab") as model_file:
    pickle.dump(per, model_file)

