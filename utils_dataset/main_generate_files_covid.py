import os

from train_val_test_files_uitls import create_train_val_test_files

if __name__ == "__main__":
    version = "5"
    save_dir = f"../records/covid_{version}"
    covid_path = "../../datasets/Dataset_Covid/COVID/Covid"
    normal_path = "../../datasets/Dataset_Covid/NONCOVID/Normal"

    # init file paths
    covid_train_file_path = os.path.join(save_dir, f"covid_train_{version}.txt")
    covid_val_file_path = os.path.join(save_dir, f"covid_val_{version}.txt")
    covid_test_file_path = os.path.join(save_dir, f"covid_test_{version}.txt")
    normal_train_file_path = os.path.join(save_dir, f"normal_train_{version}.txt")
    normal_val_file_path = os.path.join(save_dir, f"normal_val_{version}.txt")
    normal_test_file_path = os.path.join(save_dir, f"normal_test_{version}.txt")

    # covid images
    create_train_val_test_files(files_path=covid_path, train_file_path=covid_train_file_path,
                                val_file_path=covid_val_file_path, test_file_path=covid_test_file_path)

    # normal images. They have to be sub-sampled as there are many more normal images than covid images
    create_train_val_test_files(files_path=normal_path, train_file_path=normal_train_file_path,
                                val_file_path=normal_val_file_path, test_file_path=normal_test_file_path,
                                sub_sample_size=len(os.listdir(covid_path)))
