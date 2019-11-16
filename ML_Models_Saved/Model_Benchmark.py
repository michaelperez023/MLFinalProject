from ML_DataPipeline import DataPipeline
import _pickle as pickle
import pandas as pd
import glob
import os
from keras.models import load_model


path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow"
train_path = os.path.join(path, r"MLFinalProject\train_images.pickle")
test_path = os.path.join(path, r"MLFinalProject\test_images.pickle")
model_paths = glob.glob(os.path.join(path, r"MLFinalProject\ML_Models_Saved\*.pickle")) +\
                glob.glob(os.path.join(path, r"MLFinalProject\ML_Models_Saved\*.h5"))

models = dict()
accuracies = dict()
for path in model_paths:
    model_name = path.rsplit("\\", 1)[1].split(".")[0]
    if "pickle" in path:
        with open(path, "rb") as model_file:
            model = pickle.load(model_file)
    if "h5" in path:
        model = load_model(path, compile=False)
        model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
    models[model_name] = model
    accuracies[model_name] = list()

pipeline_CNN = DataPipeline("CNN", batch_size=24, partitions=12, train_file_path=train_path, test_file_path=test_path)
pipeline_SGD = DataPipeline("SGD", batch_size=24, partitions=12, train_file_path=train_path, test_file_path=test_path)
pipeline_NB = DataPipeline("NB", batch_size=24, partitions=12, train_file_path=train_path, test_file_path=test_path)
partitions = pipeline_CNN.get_partitions()

for partition in range(partitions):
    X_test_CNN, y_test_CNN = pipeline_CNN.get_test_data(partition)
    X_test_NB, y_test_NB = pipeline_NB.get_test_data(partition)
    X_test_SGD, y_test_SGD = pipeline_SGD.get_test_data(partition)
    print("Partition: %d" %partition)

    for model_name, model in models.items():
        if "CNN" in model_name:
            score = model.test_on_batch(X_test_CNN, y_test_CNN)[1]
            accuracies[model_name].append(score)
        elif "NB" in model_name:
            score = model.score(X_test_NB, y_test_NB)
            accuracies[model_name].append(score)
        else:
            score = model.score(X_test_SGD, y_test_SGD)
            accuracies[model_name].append(score)


for model_name in models.keys():
    avg = sum(accuracies[model_name])/len(accuracies[model_name])
    accuracies[model_name].append(avg)

scores = pd.DataFrame(accuracies, index=(list(range(1, partitions+1)) + ["AVG"]))
scores.to_excel("Test_Results.xlsx", sheet_name="Results", index_label="Test_Batch")
print("Testing Complete!")
print("-----------------")