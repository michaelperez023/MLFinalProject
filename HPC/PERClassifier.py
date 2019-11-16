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

labels = [0, 1]
epochs = 200
seed = 42

pipeline = DataPipeline("SGD", batch_size=96)

class_flag = True
print("Perceptron")
print("----------")
per = SGDClassifier(loss="perceptron", verbose=0, random_state=seed)
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
            per.partial_fit(x_train, y_train, classes=labels)
            class_flag = False
            continue
        per.partial_fit(x_train, y_train)

    if epoch == 0:
        with open("PERClassifier_" + str(epoch+1) + ".pickle", "ab") as model_file:
            pickle.dump(per, model_file)
    if epoch == 49:
        with open("PERClassifier_" + str(epoch+1) + ".pickle", "ab") as model_file:
            pickle.dump(per, model_file)
    if epoch == 99:
        with open("PERClassifier_" + str(epoch+1) + ".pickle", "ab") as model_file:
            pickle.dump(per, model_file)
    if epoch == (epochs - 1):
        with open("PERClassifier_" + str(epochs) + ".pickle", "ab") as model_file:
            pickle.dump(per, model_file)
