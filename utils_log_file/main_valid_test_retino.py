import glob

from facade_valid_test import valid_test

if __name__ == '__main__':
    record_dirs = glob.glob("../records/retino_*/*[!.csv]")
    for record_dir in record_dirs:
        valid_test(log_file_dir=record_dir, show_fig=False)
