from sklearn.naive_bayes import GaussianNB
from ML_DataPipeline import DataPipeline
import _pickle as pickle
import joblib
import os


path = r"C:\Users\jeffk\Desktop\MLFinalProject-master"
train_path = os.path.join(path, "train_images.pickle")
test_path = os.path.join(path, "test_images.pickle")
epochs = 10 # varied from 10-30 by 10
class_flag = True
labels = [0, 1]


pipeline = DataPipeline("NB", batch_size=24, train_file_path=train_path, test_file_path=test_path)
nb = GaussianNB()

for epoch in range(epochs):
    if (epoch % 2) == 0:
        print("Epoch %d" %(epoch+1))
    if epoch + 1 == epochs:
        print("Epoch %d" %(epochs))
    iterations = pipeline.get_iterations()
    for iters in range(iterations):
        x_train, y_train = pipeline.get_training_batch(iters)
        if class_flag:
            nb.partial_fit(x_train, y_train, classes=labels)
            class_flag = False
            continue
        nb.partial_fit(x_train, y_train)

with open("GaussianNB_" + str(epochs) + "_.pickle", "ab") as model_file:
    pickle.dump(nb, model_file)

