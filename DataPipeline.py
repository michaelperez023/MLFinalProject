"""
@Author : Peter Akioyamen
@Date : October 20, 2019
@Code : CAP 4612
"""

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

        self.train_iterations = self.train_instances // self.batch_size

    def get_iterations(self):
        """
        Returns the number of training iterations necessary to complete one epoch,
            this is predetermined by the batch size specified.
        """
        return self.iterations

    def get_training_batch(self) :
        """ """
        gray_scale = 1
        test_features = []
        test_labels = []
        return None


    def get_validation_data(self) :
        # TODO : Decide whether this will be necessary
        return None

    def get_test_data(self, test_partition) :
        """
        Retrieves the test batch to be evaluated specified by the partition number provided.

        Parameters
        ----------
        test_partition : int
            The partition number which is to be retrieved by this function

        Raises
        ------
        Exception
            If the number of partitions specified as an argument is not within
                the range of 0 and 5, inclusive
        """

        # ---
        gray_scale = 1
        test_features = []
        test_labels = []

        # ---
        if 0 <= test_partition <= 5 :
            partition = test_partition
        else :
            raise Exception("batch_count must be in range [0, 5]")

        # ---
        try :
            with open(self.train_path, "rb", ) as test_data :
                data_count = pickle.load(test_data)
                batch_interval = data_count // 6

                for i in range(data_count):
                    instance = pickle.load(test_data)
                    if (
                            (partition == 0 and i < batch_interval) or (
                            partition == 1 and batch_interval <= i < batch_interval * 2) or (
                            partition == 2 and batch_interval * 2 <= i < batch_interval * 3) or (
                            partition == 3 and batch_interval * 3 <= i < batch_interval * 4) or (
                            partition == 4 and batch_interval * 4 <= i < batch_interval * 5) or (
                            partition == 5 and i >= batch_interval * 5)
                    ) :
                        test_labels.append(instance[0])
                        test_features.append(np.asarray(instance[1]))
        except OSError as err :
            print(err.strerror)
            print("Please ensure pickle files are in correct directory.\nProgram exit.")
            quit(-1)

        # ---
        test_features = np.asarray(test_features)
        test_labels = np.asarray(test_labels)
        shape = test_features.shape
        test_features /= 255.0
        test_features = np.reshape(test_features, (shape[0], shape[1], shape[2], gray_scale))

        # ---
        return test_features, test_labels
