import os
import shutil

from train_val_test_files_uitls import split_csv_train_file

if __name__ == '__main__':
    version = "5"
    save_dir = f"../records/retino_{version}"
    os.makedirs(save_dir, exist_ok=True)

    # original files
    ori_retino_train_file = "../../datasets/Dataset_DiabeticRetinopathy/trainLabels.csv"
    ori_retino_test_file = "../../datasets/Dataset_DiabeticRetinopathy/retinopathy_solution.csv"

    # init file paths
    retino_train_file_path = os.path.join(save_dir, f"retino_train_{version}.csv")
    retino_val_file_path = os.path.join(save_dir, f"retino_val_{version}.csv")
    retino_test_file_path = os.path.join(save_dir, f"retino_test_{version}.csv")

    # create train, valid, and test files
    split_csv_train_file(
        train_file_path_ori=ori_retino_train_file,
        train_file_path=retino_train_file_path,
        val_file_path=retino_val_file_path
    )
    shutil.copy(ori_retino_test_file, retino_test_file_path)
