import os

from utils_log_file_processing import parse_log, track_training, cal_mean_std


if __name__ == '__main__':
    log_file = "../covid_b32_ncp_checkpoints/covid_log.txt"
    show_fig = True
    is_finish = True

    # init
    analysis_dir_name = "analysis"
    stat_file_name = "stats.txt"

    # create save dir at the same location with log file
    log_dir = os.path.dirname(log_file)
    analysis_dir = os.path.join(log_dir, analysis_dir_name)
    os.makedirs(os.path.join(analysis_dir), exist_ok=True)
    stat_file_path = os.path.join(analysis_dir, stat_file_name)

    # parse data
    data_dict = parse_log(log_file)

    # check training for early stop
    track_training(data_dict, include_test=is_finish, save_dir=analysis_dir, show_fig=show_fig)
    if is_finish:
        cal_mean_std(  # calculate statistics
            data_dict=data_dict,
            metrics=["[EVAL]:Acc", "[EVAL]:F1", "[EVAL]:Dice", "[TEST]:Acc", "[TEST]:F1", "[TEST]:Dice"],
            start_i=149,
            end_i=-1,
            save_file_path=stat_file_path
        )
