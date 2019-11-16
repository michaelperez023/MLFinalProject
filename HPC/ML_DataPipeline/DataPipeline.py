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
    """

    # Attempt to open and read file from the given path
    try:
        with open(file_path, "rb") as train_data:
            instance_count = pickle.load(train_data)
    except OSError as err:
        print("Please ensure pickle files are in correct directory\nProgram exit")
        raise err
    return instance_count


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
        If the current iteration number specified as an argument is not within
            the range of 0 and max_iter, inclusive
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
            If the batch_size or partitions parameters are not integer values
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
            warnings.warn("arg batch_size cannot be less than 0, default will be set")
            sys.stderr.flush()
            self.batch_size = 24
        if self.batch_size > 64:
            warnings.warn("arg batch_size larger than 64 may cause memory error, default will be set")
            sys.stderr.flush()
            self.batch_size = 24

        # Check the bounds on the set partition number
        if self.partition_count <= 8:
            warnings.warn("arg partitions less than 8 may cause memory error, default will be set")
            sys.stderr.flush()
            self.partition_count = 12
        if self.partition_count > 16:
            warnings.warn("arg partitions larger than 16 may slow down program")
            sys.stderr.flush()

        # Set the number of iterations necessary to complete one epoch
        self.train_iterations = get_instance_count(self.train_path) // self.batch_size

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

        # Define data containers, max number of iterations, and grayscale image flag
        max_iter = self.train_iterations - 1
        train_features = []
        train_labels = []
        gray_scale = 1

        # Check the current iteration
        if 0 <= iteration_num <= max_iter:
            current_iter = iteration_num
        else:
            raise ValueError("iteration_num must be in range [0, " + str(max_iter) + "]")

        # Open the training data pickle file
        try:
            with open(self.train_path, "rb") as train_data:
                # Read the number of instances in the file
                # Set the training mini-batch size
                data_count = pickle.load(train_data)
                batch_size = self.batch_size

                # Retrieve the number of training instances
                for i in range(data_count):
                    instance = pickle.load(train_data)
                    if (current_iter < max_iter) and \
                            (current_iter * batch_size <= i < (current_iter + 1) * batch_size):
                        train_labels.append(instance[0])
                        train_features.append(np.asarray(instance[1]))
                    if (current_iter == max_iter) and (current_iter * batch_size <= i):
                        train_labels.append(instance[0])
                        train_features.append(np.asarray(instance[1]))
                    if current_iter != max_iter and len(train_labels) == batch_size: break
        except OSError as err:
            # Catch OSError, notify user, exit program
            print("Ensure pickle files are in correct directory\nProgram exit")
            raise err

        # Convert the data to numpy arrays
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
            # Keep pixel values discrete
            train_features = np.reshape(train_features, (shape[0], shape[1] * shape[2]))
            train_features.astype(int)

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
            If the number of partitions specified as an argument is not within
                the range of 0 and 5, inclusive
        """

        # Define data containers, max number of partitions, and grayscale image flag
        max_partition = self.partition_count - 1
        test_features = []
        test_labels = []
        gray_scale = 1

        # Check the partition request value
        if 0 <= partition_num <= max_partition:
            current_part = partition_num
        else:
            raise ValueError("test_partition must be in range [0, 5]")

        # Open the test data pickle file
        try:
            with open(self.test_path, "rb") as test_data:
                # Read the number of test instances contained in the file
                # Compute partition size
                data_count = pickle.load(test_data)
                partition_size = data_count // (max_partition + 1)

                # Read the number of instances in a partition
                for i in range(data_count):
                    instance = pickle.load(test_data)
                    if (current_part < max_partition) and \
                            (current_part * partition_size <= i < (current_part + 1) * partition_size):
                        test_labels.append(instance[0])
                        test_features.append(np.asarray(instance[1]))
                    if (current_part == max_partition) and (current_part * partition_size <= i):
                        test_labels.append(instance[0])
                        test_features.append(np.asarray(instance[1]))
                    if current_part != max_partition and len(test_labels) == partition_size: break
        except OSError as err:
            # Catch OSError, notify user, exit program
            print("Ensure pickle files are in correct directory\nProgram exit")
            raise err

        # Convert the data to numpy arrays
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
            # Keep pixel values discrete
            test_features = np.reshape(test_features, (shape[0], shape[1] * shape[2]))
            test_features.astype(int)

        # Return features and labels for normalized test data
        return test_features, test_labels
# ----------------------------------------------------------------------------------------------------------------------
