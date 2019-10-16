import _pickle as pickle
import numpy as np

#def helper functions

class DataPipeline :
    def __init__(self, epochs, batch_size = 32, train_file_path = None, test_file_path = None):
        self.batch_size = batch_size
        self.train_instances = int #is this necessary ?
        self.test_instances = int
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

        self.iterations = self.train_instances // self.batch_size
        #self.batch_samples = np.random.randint(low = , high = , )

    def get_iterations(self):
        return self.iterations

    def get_training_batch(self) :
        return None

    def get_validation_data(self) :
        return None

    def get_test_data(self) :
        labels = []
        features = []

        try :
            with open(self.train_path, "rb", ) as test_data :
                data_count = pickle.load(test_data)
                for i in range(data_count):
                    instance = pickle.load(test_data)
                    labels.append(instance[0])
                    features.append(np.asarray(instance[1]))
        except OSError as err :
            print(err.strerror)
            print("Please ensure pickle files are in correct directory.\nProgram exit.")
            quit(-1)

        labels = np.asarray(labels)
        features = np.asarray(features)
