"""
@Author : Peter Akioyamen
@Code : CAP 4612
"""

# Import DataPipeline
from ML_DataPipeline import DataPipeline


def main() :
    # Verify Pipeline for Training Data Works #
    data_pipeline = DataPipeline()
    print("Training Methods Test")
    print("---------------------")
    train_check = []
    train_iterations = data_pipeline.get_iterations()
    # Get data in one epoch
    for iteration in range(train_iterations) :
        features, labels = data_pipeline.get_training_batch(iteration)
        if len(features) == len(labels) :
            train_check.append(True)
    print("Batch Dimensions Verification: %d" %(sum(train_check)))
    print("Training Iterations per Epoch: %d" %(train_iterations))
    if sum(train_check) == train_iterations : print("Test 1 --- Passed")
    try :
        data_pipeline = DataPipeline(train_file_path="DNE")
    except OSError :
        print("Test 2 --- Passed\n")

    # Verify Pipeline for Testing Data Works #
    data_pipeline = DataPipeline()
    print("Testing Methods Test")
    print("--------------------")
    test_partitions = data_pipeline.get_partitions()
    test_check = []
    # Get all testing data
    for partition in range(test_partitions) :
        features, labels = data_pipeline.get_test_data(partition)
        if len(features) == len(labels) :
            test_check.append(True)
    print("Testing Partitions: %d" %(test_partitions))
    print("Partition Dimensions Verification: %d" %(sum(test_check)))
    if sum(test_check) == test_partitions :
        print("Test 1 --- Passed")
    try :
        data_pipeline = DataPipeline(test_file_path = "NULL")
        for partition in range(test_partitions):
            features, labels = data_pipeline.get_test_data(partition)
            if len(features) == len(labels):
                test_check.append(True)
    except OSError :
        print("Test 2 --- Passed\n")

    # Verify Edge Cases Handled #
    print("Exception & Warning Edge Cases")
    print("------------------------------")
    print("Batch Size Tests")
    try :
        data_pipeline = DataPipeline(batch_size = 32 + 0.5)
    except ValueError :
        print("Test 1 --- Passed")
    data_pipeline = DataPipeline(batch_size = 0)
    print("Test 2 --- Passed")
    data_pipeline = DataPipeline(batch_size = 10000)
    print("Test 3 --- Passed\n")

    print("Partition Count Tests")
    try :
        data_pipeline = DataPipeline(partitions = 10 + 0.5)
    except ValueError :
        print("Test 4 --- Passed")
    data_pipeline = DataPipeline(partitions = 0)
    print("Test 5 --- Passed")
    data_pipeline = DataPipeline(partitions = 10000)
    print("Test 6 --- Passed\n")


if __name__ == '__main__':
    main()
