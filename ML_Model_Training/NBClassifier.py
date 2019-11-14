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
import os


path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow"
train_path = os.path.join(path, "train_images.pickle")
test_path = os.path.join(path, "test_images.pickle")
class_flag = True
labels = [0, 1]


pipeline = DataPipeline("NB", batch_size=24, train_file_path=train_path, test_file_path=test_path)
nb = MultinomialNB()

iterations = pipeline.get_iterations()
for iters in range(iterations):
    x_train, y_train = pipeline.get_training_batch(iters)
    if class_flag:
        nb.partial_fit(x_train, y_train, classes=labels)
        class_flag = False
        continue
    nb.partial_fit(x_train, y_train)

with open("NBClassifier.pickle", "ab") as model_file:
    pickle.dump(nb, model_file)

