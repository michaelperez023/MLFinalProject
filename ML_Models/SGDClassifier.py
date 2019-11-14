"""
@Author : Peter Akioyamen
@Code : CAP 4612

This python script is a part of the model development process.
It acts to instantiate and train 4 Support Vector Machine
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
epochs = 1 # varied from 1, 10-30 by 10
class_flag = True
labels = [0, 1]
seed = 42
cpu = joblib.cpu_count()

pipeline = DataPipeline("SGD", batch_size=24, train_file_path=train_path, test_file_path=test_path)
svm = SGDClassifier(verbose=0, n_jobs=(cpu - 2), random_state=seed)

for epoch in range(epochs):
    if (epoch % 2) == 0:
        print("Epoch %d" %(epoch+1))
    if epoch + 1 == epochs:
        print("Epoch %d" %(epochs))
    iterations = pipeline.get_iterations()
    for iters in range(iterations):
        x_train, y_train = pipeline.get_training_batch(iters)
        if class_flag:
            svm.partial_fit(x_train, y_train, classes=labels)
            class_flag = False
            continue
        svm.partial_fit(x_train, y_train)

with open("SGDClassifier_" + str(epochs) + "_.pickle", "ab") as model_file:
    pickle.dump(svm, model_file)

