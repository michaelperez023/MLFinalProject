"""
@Author : Peter Akioyamen
@Date : October 20, 2019
@Code : CAP 4612
"""


# Import libraries
import _pickle as pickle
import numpy as np


# def helper functions
class DataPipeline :
    """
    A class which facilitates the functions of a data pipeline for images.

    Attributes
    ----------
    batch_size : int
        The size of the training batches
    train_path : str
        The path, including name, designating the location of the training data pickle file
    test_path : str
        The path, including name, designating the location of the testing data pickle file
    train_iterations : int
        The number of training iterations necessary for one epoch

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """
    def __init__(self, epochs, batch_size = 24, train_file_path = None, test_file_path = None):
        """
        The constructor for the DataPipeline class.

        Parameters
        ----------
        batch_size : int, optional
            The size of the training batches (default is 24)
        train_file_path : str, optional
            The path, including name, designating the location of the training data pickle file (default is None).
                if not specified, it is assumed the file is stored in the same directory as the script
        test_file_path : str, optional
            The path, including name, designating the location of the testing data pickle file (default is None),
                if not specified, it is assumed the file is stored in the same directory as the script
        """

        # TODO : Decide whether the *_instances attribute is necessary
        self.batch_size = batch_size
        self.train_path = train_file_path
        self.test_path = test_file_path

        if train_file_path is None :
            self.train_path = "train_images.pickle"
        if test_file_path is None :
            self.test_path = "test_images.pickle"

        try :
            with open(self.train_path, "rb") as train_data, open(self.test_path, "rb") as test_data :
                self.train_instances = pickle.load(train_data)
                self.test_instances = pickle.load(test_data)
        except OSError as err :
            print(err.strerror)
            print("Please ensure pickle files are in correct directory.\nProgram exit.")
            quit(-1)

        self.train_iterations = (self.train_instances // self.batch_size) + 1  # correct computation

    def get_iterations(self):
        """
        Returns the number of training iterations necessary to complete one epoch,
            this is predetermined by the batch size specified.

        Returns
        -------
        self.iterations : int
            Number of training iterations necessary to complete one epoch
        """

        return self.train_iterations

    def get_training_batch(self, iteration_num) :
        """ """

        gray_scale = 1
        max_iter = self.train_iterations - 1
        train_features = []
        train_labels = []

        if 0 <= iteration_num <= max_iter :
            current_iter = iteration_num
        else :
            raise Exception("arg iteration_num must be in range [0, " + str(max_iter) + "]")

        try :
            with open(self.train_path, "rb") as train_data :
                data_count = pickle.load(train_data)

                for i in range(data_count) :
                    instance = pickle.load(train_data)
                    if (current_iter < max_iter) and \
                            (current_iter * self.batch_size <= i < (current_iter + 1) * self.batch_size) :
                        train_labels.append(instance[0])
                        train_features.append(np.asarray(instance[1]))
                    if (current_iter == max_iter) and (current_iter * self.batch_size <= i) :
                        train_labels.append(instance[0])
                        train_features.append(np.asarray(instance[1]))
        except OSError as err:
            # Catch OSError, notify user, exit program
            print(err.strerror)
            print("Please ensure pickle files are in correct directory.\nProgram exit.")
            quit(-1)

        # Convert the data to numpy arrays
        # Normalize the pixel values of the grayscale images
        # TODO : Randomize order of data instances
        train_features = np.asarray(train_features)
        train_labels = np.asarray(train_labels)
        shape = train_features.shape
        train_features /= 255.0
        train_features = np.reshape(train_features, (shape[0], shape[1], shape[2], gray_scale))

        # Return features and labels for normalized test data
        return train_features, train_labels

    def get_validation_data(self) :
        # TODO : Decide whether this will be necessary
        return None

    def get_test_data(self, partition_num) :
        """
        Retrieves the test batch to be evaluated specified by the partition number provided.

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
        Exception
            If the number of partitions specified as an argument is not within
                the range of 0 and 5, inclusive
        """

        # TODO : Check math on the partitioning here
        # Define data containers and grayscale image flag
        gray_scale = 1
        max_partition = 5
        test_features = []
        test_labels = []

        # Check the partition request value
        if 0 <= partition_num <= max_partition :
            partition = partition_num
        else :
            raise Exception("arg test_partition must be in range [0, 5]")

        # Open the training data pickle file
        try :
            with open(self.test_path, "rb") as test_data :
                # Read the number of instances in the file, compute partition size
                data_count = pickle.load(test_data)
                partition_size = data_count // (max_partition + 1)

                # Read the number of instances in a partition
                for i in range(data_count):
                    instance = pickle.load(test_data)
                    if (partition < max_partition) and \
                            (partition * partition_size <= i < (partition + 1) * partition_size) :
                        test_labels.append(instance[0])
                        test_features.append(np.asarray(instance[1]))
                    if (partition == max_partition) and (partition * partition_size <= i) :
                        test_labels.append(instance[0])
                        test_features.append(np.asarray(instance[1]))
        except OSError as err :
            # Catch OSError, notify user, exit program
            print(err.strerror)
            print("Please ensure pickle files are in correct directory.\nProgram exit.")
            quit(-1)

        # Convert the data to numpy arrays
        # Normalize the pixel values of the grayscale images
        test_features = np.asarray(test_features)
        test_labels = np.asarray(test_labels)
        shape = test_features.shape
        test_features /= 255.0
        test_features = np.reshape(test_features, (shape[0], shape[1], shape[2], gray_scale))

        # Return features and labels for normalized test data
        return test_features, test_labels
