import sys

sys.path.append("./utils_log_file_processing")  # root dir contains main files

from .parse_log import parse_log
from .valid_test import track_training, get_stats
