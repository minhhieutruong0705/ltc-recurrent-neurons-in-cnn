import os
import numpy as np


def create_train_val_test_files(files_path, train_file_path, val_file_path, test_file_path, sub_sample_size=None):
    filenames = os.listdir(files_path)
    np.random.shuffle(filenames)
    if sub_sample_size is not None:
        filenames = filenames[:sub_sample_size]
    n_files = len(filenames)

    # Create train.txt
    with open(train_file_path, "w") as f:
        for filename in filenames[0:int(0.65 * n_files)]:
            f.write(filename + '\n')

    # Create valid.txt
    with open(val_file_path, "w") as f:
        for filename in filenames[int(0.65 * n_files):int(0.85 * n_files)]:
            f.write(filename + '\n')

    # Create test.txt
    with open(test_file_path, "w") as f:
        for filename in filenames[int(0.85 * n_files):]:
            f.write(filename + '\n')
