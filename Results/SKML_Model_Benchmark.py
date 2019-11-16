from ML_DataPipeline import DataPipeline
import _pickle as pickle
import pandas as pd
import glob
import os

path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow"
train_path = os.path.join(path, r"MLFinalProject\train_images.pickle")
test_path = os.path.join(path, r"MLFinalProject\test_images.pickle")
sklearn_paths = glob.glob(os.path.join(path, r"MLFinalProject\ML_Model_Training\*.pickle"))

pipeline_SGD = DataPipeline("SGD", batch_size=24, partitions=16, train_file_path=train_path, test_file_path=test_path)
pipeline_NB = DataPipeline("NB", batch_size=24, partitions=16, train_file_path=train_path, test_file_path=test_path)

sklearn_models = ["LOG", "NB", "PER", "SVM"]
models = dict()

for index, sklearn_path in enumerate(sklearn_paths):
    with open(sklearn_path, "rb") as model_file:
        model = pickle.load(model_file)
        models[sklearn_models[index]] = model


model_keys = sklearn_models
accuracies = dict()
for key in model_keys :
    accuracies[key] = []

partitions = pipeline_NB.get_partitions()
for partition in range(partitions):
    X_test_NB, y_test_NB = pipeline_NB.get_test_data(partition)
    X_test_SGD, y_test_SGD = pipeline_SGD.get_test_data(partition)

    for key in model_keys:
        if "NB" in key:
            accuracy = models[key].score(X_test_NB, y_test_NB)
            print(key)
            print("Accuracy: %f" % accuracy)
            accuracies[key].append(accuracy)
        else:
            accuracy = models[key].score(X_test_SGD, y_test_SGD)
            print(key)
            print("Accuracy: %f" % accuracy)
            accuracies[key].append(accuracy)


for key in model_keys:
    avg = sum(accuracies[key])/len(accuracies[key])
    accuracies[key].append(avg)

scores = pd.DataFrame(accuracies, index=(list(range(1, partitions+1)) + ["AVG"]))
scores.to_excel("results.xlsx", sheet_name="Results", index_label="Test_Batch")