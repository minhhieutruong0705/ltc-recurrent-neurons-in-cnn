import os
import numpy as np
import csv


def create_train_val_test_files(files_path, train_file_path, val_file_path, test_file_path, sub_sample_size=None):
    filenames = os.listdir(files_path)
    np.random.shuffle(filenames)
    if sub_sample_size is not None:
        filenames = filenames[:sub_sample_size]
    n_files = len(filenames)

    # create train.txt
    with open(train_file_path, "w") as f:
        for filename in filenames[0:int(0.65 * n_files)]:
            f.write(filename + '\n')

    # create valid.txt
    with open(val_file_path, "w") as f:
        for filename in filenames[int(0.65 * n_files):int(0.85 * n_files)]:
            f.write(filename + '\n')

    # create test.txt
    with open(test_file_path, "w") as f:
        for filename in filenames[int(0.85 * n_files):]:
            f.write(filename + '\n')


def split_csv_train_file(train_file_path_ori, train_file_path, val_file_path):
    lines = []
    # read csv file
    with open(train_file_path_ori, 'r') as f:
        csv_reader = csv.DictReader(f)
        for line in csv_reader:
            lines.append(line)
    # shuffle
    np.random.shuffle(lines)

    # params
    fieldnames = lines[0].keys()
    file_length = len(lines)

    # create train.txt
    with open(train_file_path, 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(lines[:int(0.75 * file_length)])

    # create valid.txt
    with open(val_file_path, 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(lines[int(0.75 * file_length):])
