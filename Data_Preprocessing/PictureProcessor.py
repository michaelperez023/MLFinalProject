"""
@Author : Peter Akioyamen
@Code : CAP 4612

This python script is a part of the early stage data pipeline.
The data are chest XRay images of varying dimensionality. This
script reads images from local disk, down-samples them, and serializes
the images and their labels for storage. 0 denotes pneumonia, 1 is normal.

Usage: python Data_Preprocessing/PictureProcessor.py --dataset-dir path/to/chest_xray --out-dir path/to/output
"""


# Define main function of script
def main(argv=None):
    # Import libraries
    from PIL import Image
    import _pickle as pickle
    import numpy as np
    import argparse
    import collections
    import glob
    import os

    # Read the command-line options
    parser = argparse.ArgumentParser(description="Resize the chest X-ray images to grayscale squares and save them, "
                                                 "with their labels, as train_images.pickle and test_images.pickle.")
    parser.add_argument("--dataset-dir", required=True,
                        help="the dataset's chest_xray folder, which holds train/ and test/, each with NORMAL/ "
                             "and PNEUMONIA/")
    parser.add_argument("--out-dir", default=".", help="folder for the two pickle files (default: current folder)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for shuffling the image order (default: unseeded, as in 2019)")
    parser.add_argument("--size", type=int, default=750, help="side of the resized images in pixels (default: 750)")
    args = parser.parse_args(argv)

    # Define local file paths to image directories
    train_normal_path = os.path.join(args.dataset_dir, "train", "NORMAL")
    train_pneumonia_path = os.path.join(args.dataset_dir, "train", "PNEUMONIA")
    test_normal_path = os.path.join(args.dataset_dir, "test", "NORMAL")
    test_pneumonia_path = os.path.join(args.dataset_dir, "test", "PNEUMONIA")
    image_paths = {"train_normal": os.path.join(train_normal_path, "*.jpeg"),
                   "train_pneumonia": os.path.join(train_pneumonia_path, "*.jpeg"),
                   "test_normal": os.path.join(test_normal_path, "*.jpeg"),
                   "test_pneumonia": os.path.join(test_pneumonia_path, "*.jpeg")}

    # Retrieve the full paths of the images including file names and extensions
    # Sort them so that a seeded shuffle gives the same order on every machine
    # Get the number of images and define the image resizing dimensions
    image_files = {key: sorted(glob.glob(path)) for key, path in image_paths.items()}
    empty = [image_paths[key] for key, files in image_files.items() if not files]
    if empty:
        raise SystemExit("No images match:\n  " + "\n  ".join(empty) + "\n--dataset-dir must be the dataset's "
                         "chest_xray folder, which holds train/ and test/, each with NORMAL/ and PNEUMONIA/")
    image_files_keys = list(image_files.keys())
    count_images = [sum([len(image_files[image_files_keys[0]]), len(image_files[image_files_keys[1]])]),
                    sum([len(image_files[image_files_keys[2]]), len(image_files[image_files_keys[3]])])]
    image_key_pairs = []
    for key in image_files_keys:
        image_files_list = [[key, file] for file in image_files[key]]
        image_key_pairs.extend(image_files_list)
    if args.seed is None:
        np.random.shuffle(image_key_pairs)
    else:
        np.random.RandomState(args.seed).shuffle(image_key_pairs)

    # Dimensions decided based on heuristics, image dimension averages, and memory constraints
    dim = (args.size, args.size)
    # Nearest-neighbour resampling was the default of Pillow 6, used in 2019; Pillow 7 changed the default
    nearest = getattr(Image, "Resampling", Image).NEAREST

    # Save the number of test and training images
    # Open each image file, resize it, encode into binary and save it to a pickle file
    # Generate and save the labels of the images
    # Write to temporary files first, so that an interrupted run leaves no incomplete dataset behind
    os.makedirs(args.out_dir, exist_ok=True)
    train_file = os.path.join(args.out_dir, "train_images.pickle")
    test_file = os.path.join(args.out_dir, "test_images.pickle")
    modes = collections.Counter()
    with open(train_file + ".tmp", "wb") as train_f, open(test_file + ".tmp", "wb") as test_f:
        pickle.dump(count_images[0], train_f)
        pickle.dump(count_images[1], test_f)
        for pair in image_key_pairs:
            key, image = pair
            with Image.open(image) as source:
                modes[source.mode] += 1
                im = source.convert("L").resize(dim, nearest)
            if key == image_files_keys[0] :
                pickle.dump([1, im], train_f)
            elif key == image_files_keys[1] :
                pickle.dump([0, im], train_f)
            elif key == image_files_keys[2] :
                pickle.dump([1, im], test_f)
            else :
                pickle.dump([0, im], test_f)
            im = None
    os.replace(train_file + ".tmp", train_file)
    os.replace(test_file + ".tmp", test_file)

    # Notify user of successful execution
    print("Success: %d training and %d test images saved in %s" % (count_images[0], count_images[1], args.out_dir))
    for key in image_files_keys:
        print("  %-16s %5d" % (key, len(image_files[key])))
    print("  image modes before conversion to grayscale: "
          + ", ".join("%s %d" % (mode, n) for mode, n in sorted(modes.items())))


if __name__ == '__main__':
    main()
