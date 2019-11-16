"""
@Author : Peter Akioyamen
@Code : CAP 4612

This python script is a part of the early stage data pipeline.
The data are chest XRay images of varying dimensionality. This
script reads images from local disk, down-samples them, and serializes
the images and their labels for storage. 0 denotes pneumonia, 1 is normal.
"""

# Define main function of script
def main():
    # Import libraries
    from PIL import Image
    import _pickle as pickle
    import numpy as np
    import glob
    import os

    # Define local file paths to image directories
    train_normal_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\train\NORMAL"
    train_pneumonia_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\train\PNEUMONIA"
    test_normal_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\test\NORMAL"
    test_pneumonia_path = r"C:\Users\Peter\Documents\UWO Work\Year 3\Sem1\CS Machine Learning (POLY)\Project Marrow\DataFiles\chest_xray\test\PNEUMONIA"
    image_paths = {"train_normal": os.path.join(train_normal_path, "*.jpeg"),
                   "train_pneumonia": os.path.join(train_pneumonia_path, "*.jpeg"),
                   "test_normal": os.path.join(test_normal_path, "*.jpeg"),
                   "test_pneumonia": os.path.join(test_pneumonia_path, "*.jpeg")}

    # Retrieve the full paths of the images including file names and extensions
    # Get the number of images and define the image resizing dimensions
    image_files = {key: glob.glob(path) for key, path in image_paths.items()}
    image_files_keys = list(image_files.keys())
    count_images = [sum([len(image_files[image_files_keys[0]]), len(image_files[image_files_keys[1]])]),
                    sum([len(image_files[image_files_keys[2]]), len(image_files[image_files_keys[3]])])]
    image_key_pairs = []
    for key in image_files_keys:
        image_files_list = [[key, file] for file in image_files[key]]
        image_key_pairs.extend(image_files_list)
    np.random.shuffle(image_key_pairs)

    # Dimensions decided based on heuristics, image dimension averages, and memory constraints
    dim = [750, 750]

    # Save the number of test and training images
    # Open each image file, resize it, encode into binary and save it to a pickle file
    # Generate and save the labels of the images
    with open("train_images.pickle", "ab+") as train_f, open("test_images.pickle", "ab+") as test_f:
        pickle.dump(count_images[0], train_f)
        pickle.dump(count_images[1], test_f)
        for pair in image_key_pairs:
            key, image = pair
            with Image.open(image).convert("L") as im:
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

        # Notify user of successful execution
        print("Success")


if __name__ == '__main__':
    main()
