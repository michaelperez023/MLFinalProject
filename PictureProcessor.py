"""
@Author : Peter Akioyamen
@Date : October 10, 2019
@Code : CAP 4612

This python script is a part of the early stage data pipeline.
The data are chest XRay images of varying dimensionality. This
script reads images from local disk, formats them, and serializes
the images and their labels for storage.
"""

from PIL import Image
import _pickle as pickle
import glob
import os

train_normal_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\train\NORMAL"
train_pneumonia_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\train\PNEUMONIA"
test_normal_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\test\NORMAL"
test_pneumonia_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\test\PNEUMONIA"
image_paths = {"train_normal": os.path.join(train_normal_path, "*.jpeg"),
               "train_pneumonia": os.path.join(train_pneumonia_path, "*.jpeg"),
               "test_normal": os.path.join(test_normal_path, "*.jpeg"),
               "test_pneumonia": os.path.join(test_pneumonia_path, "*.jpeg")}
image_files = {key: glob.glob(path) for key, path in image_paths.items()}
image_files_keys = list(image_files.keys())
count_images = [sum([len(image_files[image_files_keys[0]]), len(image_files[image_files_keys[1]])]),
                sum([len(image_files[image_files_keys[2]]), len(image_files[image_files_keys[3]])])]
dim = [1000, 1000]

with open("../train_images.pickle", "ab+") as train_f, open("../test_images.pickle", "ab+") as test_f :
    pickle.dump(count_images[0], train_f)
    pickle.dump(count_images[1], test_f)
    for key, image_category in image_files.items() :
        for image in image_category :
            with Image.open(image) as im :
                im = im.resize(dim)
                if key == image_files_keys[0] :
                    pickle.dump([1, im], train_f)
                elif key == image_files_keys[1] :
                    pickle.dump([0, im], train_f)
                elif key == image_files_keys[2] :
                    pickle.dump([1, im], test_f)
                else :
                    pickle.dump([0, im], test_f)
            im = None
    print("Success")
