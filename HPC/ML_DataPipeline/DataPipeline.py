"""
@Author : Peter Akioyamen
@Code : CAP 4612
"""

# Import standard libraries
import sys
import warnings
import _pickle as pickle

# Import third part libraries
import numpy as np


# Helper Functions
# ------------------------------------------------------------------------------------------------------------------
def missing_file_message(file_path):
    """
    Returns the error message for a pickle file that does not exist.
    """

    return ("pickle file not found: " + str(file_path) + "\nCreate train_images.pickle and test_images.pickle "
            "with Data_Preprocessing/PictureProcessor.py and pass their location (see README)")


def check_instance_count(file_path, instance_count):
    """
    Raises a ValueError unless instance_count, the first object of a pickle file, is a positive integer.
    """

    if not isinstance(instance_count, int) or instance_count <= 0:
        raise ValueError(str(file_path) + " should start with the number of images, found " + repr(instance_count)
                         + "; run PictureProcessor.py again")


def get_instance_count(file_path):
    """
    Opens a pickle file specified by file_path; reads the first object which
        should be an integer representing the number of object instances in the file.

    Parameters
    ----------
    file_path : str
        The path, including name, designating the location of a pickle file

    Returns
    -------
    instance_count : int
        The number of object instances stored in the file

    Raises
    ------
    FileNotFoundError
        If there is no file at file_path
    ValueError
        If the file is empty or does not start with a pickled object
    """

    # Attempt to open and read file from the given path
    try:
        with open(file_path, "rb") as train_data:
            instance_count = pickle.load(train_data)
    except FileNotFoundError as err:
        raise FileNotFoundError(missing_file_message(file_path)) from err
    except (EOFError, pickle.UnpicklingError):
        raise ValueError(str(file_path) + " is empty or not a file written by PictureProcessor.py") from None
    return instance_count


def load_images(file_path):
    """
    Reads a pickle file written by PictureProcessor.py from start to end, once.

    The file holds an integer count followed by that many [label, image] pairs,
        where the image is a grayscale PIL image and the label is 0 (pneumonia) or 1 (normal).

    Parameters
    ----------
    file_path : str
        The path, including name, designating the location of a pickle file

    Returns
    -------
    images : ndarray
        A uint8 numpy array of shape (count, height, width) holding the pixel values
    labels : ndarray
        A numpy array of shape (count,) holding the labels

    Raises
    ------
    FileNotFoundError
        If there is no file at file_path
    ValueError
        If the file does not have the format written by PictureProcessor.py
    """

    try:
        data_file = open(file_path, "rb")
    except FileNotFoundError as err:
        raise FileNotFoundError(missing_file_message(file_path)) from err

    with data_file:
        try:
            data_count = pickle.load(data_file)
        except (EOFError, pickle.UnpicklingError):
            raise ValueError(str(file_path) + " is empty or not a file written by PictureProcessor.py") from None
        check_instance_count(file_path, data_count)
        images = None
        labels = np.empty(data_count, dtype=np.int64)
        for i in range(data_count):
            try:
                label, image = pickle.load(data_file)
            except (EOFError, pickle.UnpicklingError):
                raise ValueError(str(file_path) + " is incomplete: it holds " + str(i) + " of the " + str(data_count)
                                 + " images its first entry announces; run PictureProcessor.py again") from None
            pixels = np.asarray(image)
            if images is None:
                if pixels.ndim != 2 or pixels.dtype != np.uint8:
                    raise ValueError(str(file_path) + " should hold 8-bit grayscale images, found an array of shape "
                                     + str(pixels.shape) + " and type " + str(pixels.dtype))
                images = np.empty((data_count,) + pixels.shape, dtype=np.uint8)
            if pixels.shape != images.shape[1:]:
                raise ValueError("image " + str(i) + " in " + str(file_path) + " has shape " + str(pixels.shape)
                                 + ", the first image has shape " + str(images.shape[1:]))
            images[i] = pixels
            labels[i] = label
        if data_file.read(1):
            # The 2019 version of PictureProcessor.py opened its output files in append mode,
            # so running it twice stacked two datasets in one file
            raise ValueError(str(file_path) + " has more data after its " + str(data_count) + " images; it was "
                             "probably written twice. Delete it and run PictureProcessor.py again")
    return images, labels


def shuffle_data(features_array, labels_array):
    """
    Checks the number of instances in both arrays and randomizes the order of instances.

    Parameters
    ----------
    features_array : ndarray
        A numpy array containing the features of each instance
    labels_array : ndarray
        A numpy array containing the labels of each instance

    Returns
    -------
    permuted_features : ndarray
        A numpy array containing the instances' features in randomized order
    permuted_labels : ndarray
        A numpy array containing the instances' labels in randomized order

    Raises
    ------
    ValueError
        If the two arrays do not hold the same number of instances
    """

    # Ensure the number of instances is the same
    if features_array.shape[0] == labels_array.shape[0]:
        # Permute the data
        permute = np.random.permutation(labels_array.shape[0])
        permuted_features = features_array[permute]
        permuted_labels = labels_array[permute]

        # Return permuted data
        return permuted_features, permuted_labels
    else:
        raise ValueError("shape mismatch: arrays must have same number of instances")
# ------------------------------------------------------------------------------------------------------------------


# Classes
# ----------------------------------------------------------------------------------------------------------------------
class DataPipeline:
    """
    A class which facilitates the functions of a data pipeline for images.

    Each pickle file is read once, the first time its data are requested, and kept in memory
        as 8-bit pixels (about 2.9 GB for the 5,232 training images at 750x750, 0.35 GB for the 624 test images).
        Until 2026 every request re-read the file from its start.

    Attributes
    ----------
    ml_model : str
        The short-name of the model to be trained
    batch_size : int
        The size of the training batches
    train_path : str
        The path, including name, designating the location of the training data pickle file
    test_path : str
        The path, including name, designating the location of the testing data pickle file
    train_iterations : int
        The number of training iterations necessary for one epoch
    partition_count : int
        The number of partitions necessary to retrieve all test data
    train_data : tuple of ndarray or None
        The training images and labels, once read
    test_data : tuple of ndarray or None
        The test images and labels, once read

    Methods
    -------
    get_iterations()
        Returns the number of iterations needed to complete an epoch
    get_partitions()
        Returns the number of partitions needed to retrieve all test data
    get_training_batch(iteration_num)
        Returns a mini-batch of training data that has been randomized
    get_test_data(partition_num)
        Returns the subset of the test data depending on the specified partition
    """
    def __init__(self, model, batch_size=24, partitions=12, train_file_path=None, test_file_path=None):
        """
        The constructor for the DataPipeline class.

        Parameters
        ----------
        model : str
            The short-name of the model to be trained; One of "NB", "SGD", or "CNN" for
            Naive Bayes Classifier, Stochastic Gradient Descent Classifier, or Convolution Neural Network
        batch_size : int, optional
            The size of the training batches (default is 24)
        partitions : int, optional
            The number of partitions of the test set (default is 12)
        train_file_path : str, optional
            The path, including name, designating the location of the training data pickle file (default is None),
                if not specified, it is assumed the file is stored in the same directory as the script
        test_file_path : str, optional
            The path, including name, designating the location of the testing data pickle file (default is None),
                if not specified, it is assumed the file is stored in the same directory as the script

        Raises
        ------
        ValueError
            If the batch_size or partitions parameters are not integer values, or if the training data pickle
                file does not start with a positive image count or holds fewer images than one batch
        FileNotFoundError
            If there is no training data pickle file at train_file_path
        """

        # Set the type of machine learning model
        if model not in ["NB", "SGD", "CNN"]:
            raise ValueError("model must be one of NB, SGD, or CNN")
        else:
            self.ml_model = model

        # Set the location of the training and test data file to the current directory
        if train_file_path is None:
            self.train_path = "train_images.pickle"
        else:
            self.train_path = train_file_path
        if test_file_path is None:
            self.test_path = "test_images.pickle"
        else:
            self.test_path = test_file_path

        # Set the batch size for training
        if isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            raise ValueError("batch_size must be an integer value")

        # Set the number of partitions for the training data
        if isinstance(partitions, int):
            self.partition_count = partitions
        else:
            raise ValueError("partition_count must be an integer value")

        # Check the bounds on the set batch size
        if self.batch_size <= 0:
            warnings.warn("arg batch_size must be larger than 0, default will be set")
            sys.stderr.flush()
            self.batch_size = 24
        if self.batch_size > 64:
            warnings.warn("arg batch_size larger than 64 may cause memory error, default will be set")
            sys.stderr.flush()
            self.batch_size = 24

        # Check the bounds on the set partition number
        if self.partition_count <= 8:
            warnings.warn("arg partitions of 8 or fewer may cause memory error, default will be set")
            sys.stderr.flush()
            self.partition_count = 12
        if self.partition_count > 16:
            warnings.warn("arg partitions larger than 16 may slow down program")
            sys.stderr.flush()

        # Set the number of iterations necessary to complete one epoch
        # Without a full batch there would be no iterations, and the training scripts would save untrained models
        train_count = get_instance_count(self.train_path)
        check_instance_count(self.train_path, train_count)
        if train_count < self.batch_size:
            raise ValueError(str(self.train_path) + " holds " + str(train_count) + " images, fewer than one batch of "
                             + str(self.batch_size))
        self.train_iterations = train_count // self.batch_size

        # The images are read from disk the first time they are needed, then kept in memory
        self.train_data = None
        self.test_data = None

    def get_iterations(self):
        """
        Returns the number of training iterations necessary to complete one epoch,
            this is predetermined by the batch size specified.

        Returns
        -------
        self.train_iterations : int
            Number of training iterations necessary to complete one epoch
        """

        return self.train_iterations

    def get_partitions(self):
        """
        Returns the number of partitions necessary to retrieve all test data,
            this is predetermined due to estimated memory constraints.

        Returns
        -------
        self.test_partitions : int
            Number of test data partitions necessary to retrieve all the data
        """

        return self.partition_count

    def get_training_batch(self, iteration_num):
        """
        Retrieves and randomizes a mini-batch of data for training in the current iteration.

        Parameters
        ----------
        iteration_num : int
            The current training iteration in this epoch

        Returns
        -------
        train_features : ndarray
            A numpy array containing the features of each training data instance within the batch
        train_labels : ndarray
            A numpy array containing the labels of the training data for the batch

        Raises
        ------
        ValueError
            If the current iteration number specified as an argument is not within
                the range of 0 and max_iter, inclusive
        """

        # Define max number of iterations and grayscale image flag
        max_iter = self.train_iterations - 1
        gray_scale = 1

        # Check the current iteration
        if 0 <= iteration_num <= max_iter:
            current_iter = iteration_num
        else:
            raise ValueError("iteration_num must be in range [0, " + str(max_iter) + "]")

        # Read the training data pickle file on the first request
        if self.train_data is None:
            self.train_data = load_images(self.train_path)
        images, labels = self.train_data

        # Select the mini-batch; the last iteration of an epoch also takes the remaining instances
        start = current_iter * self.batch_size
        stop = start + self.batch_size if current_iter < max_iter else len(labels)
        train_features = images[start:stop]
        train_labels = labels[start:stop]

        # Convert the data to floating point numpy arrays
        # Randomize the order of the training data
        train_features = np.asarray(train_features, dtype=float)
        train_labels = np.asarray(train_labels)
        train_features, train_labels = shuffle_data(train_features, train_labels)
        shape = train_features.shape

        # If model is CNN
        if self.ml_model == "CNN" :
            # Reshape the data to instances x image dim (mxn) x 1
            # Normalize the pixel values of the grayscale images
            train_features = np.reshape(train_features, (shape[0], shape[1], shape[2], gray_scale))
            train_features /= 255.0
        # If model is SGD
        elif self.ml_model == "SGD":
            # Reshape the data to instances x pixels
            # Standardize pixel values of grayscale image
            train_features = np.reshape(train_features, (shape[0], shape[1] * shape[2]))
            train_mean = train_features.mean(axis=1, keepdims=True)
            train_std = train_features.std(axis=1, keepdims=True)
            train_features = (train_features - train_mean) / train_std
        else:
            # Reshape the data to instances x pixels
            # Keep pixel values discrete: whole numbers from 0 to 255
            train_features = np.reshape(train_features, (shape[0], shape[1] * shape[2]))

        # Return features and labels for the train data
        return train_features, train_labels

    def get_test_data(self, partition_num):
        """
        Retrieves the test batch to be evaluated, specified by the partition number provided.

        Parameters
        ----------
        partition_num : int
            The partition number which is to be retrieved by this function

        Returns
        -------
        test_features : ndarray
            A numpy array containing the features of each test data instance in a partition
        test_labels : ndarray
            A numpy array containing the labels of the test data in a partition

        Raises
        ------
        ValueError
            If the partition number specified as an argument is not within
                the range of 0 and the number of partitions - 1, inclusive
        """

        # Define max number of partitions and grayscale image flag
        max_partition = self.partition_count - 1
        gray_scale = 1

        # Check the partition request value
        if 0 <= partition_num <= max_partition:
            current_part = partition_num
        else:
            raise ValueError("partition_num must be in range [0, " + str(max_partition) + "]")

        # Read the test data pickle file on the first request
        if self.test_data is None:
            self.test_data = load_images(self.test_path)
        images, labels = self.test_data

        # Compute partition size and select the partition; the last partition also takes the remaining instances
        partition_size = len(labels) // (max_partition + 1)
        start = current_part * partition_size
        stop = start + partition_size if current_part < max_partition else len(labels)
        test_features = images[start:stop]
        test_labels = labels[start:stop]

        # Convert the data to floating point numpy arrays
        test_features = np.asarray(test_features, dtype=float)
        test_labels = np.asarray(test_labels)
        shape = test_features.shape

        # If model is CNN
        if self.ml_model == "CNN" :
            # Reshape the data to instances x image dim (mxn) x 1
            # Normalize the pixel values of the grayscale images
            test_features = np.reshape(test_features, (shape[0], shape[1], shape[2], gray_scale))
            test_features /= 255.0
        # If model is SGD
        elif self.ml_model == "SGD":
            # Reshape the data to instances x pixels
            # Standardize pixel values of grayscale image
            test_features = np.reshape(test_features, (shape[0], shape[1] * shape[2]))
            test_mean = test_features.mean(axis=1, keepdims=True)
            test_std = test_features.std(axis=1, keepdims=True)
            test_features = (test_features - test_mean) / test_std
        else:
            # Reshape the data to instances x pixels
            # Keep pixel values discrete: whole numbers from 0 to 255
            test_features = np.reshape(test_features, (shape[0], shape[1] * shape[2]))

        # Return features and labels for normalized test data
        return test_features, test_labels
# ----------------------------------------------------------------------------------------------------------------------
