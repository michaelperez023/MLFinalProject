import os
import glob
import numpy as np
from PIL import Image

train_normal_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\train\NORMAL"
train_pneumonia_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\train\PNEUMONIA"
test_normal_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\test\NORMAL"
test_pneumonia_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\test\PNEUMONIA"
image_paths = [os.path.join(train_normal_path, "*.jpeg"),
               os.path.join(train_pneumonia_path, "*.jpeg"),
               os.path.join(test_normal_path, "*.jpeg"),
               os.path.join(test_pneumonia_path, "*.jpeg")]
image_files = [glob.glob(path) for path in image_paths]
images_dict = {"train_normal": [],
          "train_pneumonia": [],
          "test_normal": [],
          "test_pneumonia": []}
images_dict_keys = list(images_dict.keys())
images_dim = []
json = {"Train": {1: [], 0: []}, "Test": {1: [], 0: []}}

for index, data_category in enumerate(images_dict) :
    for image in image_files[index] :
        im = Image.open(image)
        images_dict[data_category].append(im)
        images_dim.append(np.array(im.size))

images_dim = np.asarray(images_dim)
print(images_dict_keys[0], len(images_dict[images_dict_keys[0]]))
print(images_dict_keys[1], len(images_dict[images_dict_keys[1]]))
print(images_dict_keys[2], len(images_dict[images_dict_keys[2]]))
print(images_dict_keys[3], len(images_dict[images_dict_keys[3]]))
print(len(images_dim))

sorted1 = np.sort(images_dim[:, 0])
sorted2 = np.sort(images_dim[:, 1])
new_dim = [int(np.mean((sorted1[0], sorted1[-1]))), int(np.mean((sorted2[0], sorted2[-1])))]

print(np.asarray(images_dict[images_dict_keys[3]][0]).shape)

for key in images_dict_keys :
    for index, im in enumerate(images_dict[key]) :
        print(index)
        new_image = np.asarray(im.resize(new_dim))
        if "train" in key and "normal" in key :
            json["Train"][1].append(new_image)
        elif "train" in key and "pneumonia" in key :
            json["Train"][0].append(new_image)
        elif "test" in key and "normal" in key :
            json["Test"][1].append(new_image)
        else :
            json["Test"][0].append(new_image)

print(json)
print(images_dict_keys[0], len(images_dict[images_dict_keys[0]]))
print(images_dict_keys[1], len(images_dict[images_dict_keys[1]]))
print(images_dict_keys[2], len(images_dict[images_dict_keys[2]]))
print(images_dict_keys[3], len(images_dict[images_dict_keys[3]]))


"""
test_ = glob.glob(image_paths[0])
im = Image.open(test_[2])
print(im.size)
im.show()
im.resize((500, 500))
"""
