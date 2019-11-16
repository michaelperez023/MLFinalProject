from ML_DataPipeline import DataPipeline
from PIL import Image
import _pickle as pickle
import pandas as pd
import numpy as np
import os


path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow"
train_path = os.path.join(path, "train_images.pickle")
test_path = os.path.join(path, "test_images.pickle")
nb_path = os.path.join(path, r"MLFinalProject\ML_Models_Saved\NBClassifier.pickle")

with open(nb_path, "rb") as model :
    nb_classifier = pickle.load(model)
pipeline = DataPipeline("NB", train_file_path=train_path, test_file_path=test_path)
partitions = pipeline.get_partitions()

random_batch = np.random.randint(0, partitions-1)
X_, y_ = pipeline.get_test_data(random_batch)

sample_image = X_[0].reshape((750, 750))
sample_label = y_[0]

y_hat = nb_classifier.predict([X_[0]])
accuracy = nb_classifier.score(X_, y_)
results = pd.DataFrame({"Prediction": y_hat, "Actual": y_[0]})

print("Project Marrow Beta Demo: Multinomial Naive Bayes")
print("-------------------------------------------------")
if sample_label == 0: print("Sample Image Displays Pneumonia")
else: print("Sample Image Displays Normal")
im = Image.fromarray(sample_image)
im.show()
input("\n")
print("\nPrediction on Sample Image")
print(results)
print("Benchmark Prediction on Test Subset: %f" %accuracy)


