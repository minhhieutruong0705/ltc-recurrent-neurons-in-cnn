import os

from utils_log_file_processing import parse_log, track_training, get_stats

if __name__ == '__main__':
    checkpoint_dir = "../covid_b32_ncp_checkpoints"
    show_fig = True
    is_finish = True

    # get training name and log file path
    training_name = os.path.basename(checkpoint_dir).replace('_checkpoints', '')
    log_file = os.path.join(checkpoint_dir, f"{training_name}_log.txt")

    # init save dir and files
    analysis_dir_name = f"{training_name}_analysis"
    stat_file_name = f"{training_name}_stats.txt"

    # create save dir in checkpoint dir
    analysis_dir = os.path.join(checkpoint_dir, analysis_dir_name)
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
    if is_finish:
        get_stats(data_dict=data_dict,
                  metrics=["[VALID]:Acc", "[VALID]:F1", "[VALID]:Dice"],
                  start_i=150, end_i=-1, save_file_path=stat_file_path)
