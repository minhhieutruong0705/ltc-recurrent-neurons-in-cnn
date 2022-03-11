import os

from utils_log_file_processing import parse_log, track_training, get_stats

if __name__ == '__main__':
    shuffler_version = 1
    record_dir = "records/covid_1/covid_crnet_1"
    show_fig = True

    # get training name and log file path
    training_name = os.path.basename(record_dir)
    log_file = os.path.join(record_dir, f"{training_name}_log.txt")

    # init save dir and files
    analysis_dir_name = f"{training_name}_analysis"
    stat_file_name = f"{training_name}_stats.txt"

    # create save dir in checkpoint dir
    analysis_dir = os.path.join(record_dir, analysis_dir_name)
    os.makedirs(os.path.join(analysis_dir), exist_ok=True)
    stat_file_path = os.path.join(analysis_dir, stat_file_name)

    # parse data
    data_dict = parse_log(log_file)

    # check training for early stop
    track_training(
        data_dict,
        training_name=training_name,
        save_dir=analysis_dir,
        show_fig=show_fig
    )

    # calculate stats
    get_stats(data_dict=data_dict,
              metrics=["[VALID]:Acc", "[VALID]:F1", "[VALID]:Dice"],
              start_i=150, end_i=-1, save_file_path=stat_file_path)
