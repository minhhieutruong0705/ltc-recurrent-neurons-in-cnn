import math

import matplotlib.pyplot as plt
import numpy as np
import os


def __parse_data__(log_file):
    # init data holders
    train_data = {"Loss": [], "Acc": [], "F1": [], "Dice": [], "Pre": [], "Re": [], "TP": [], "TN": [], "FP": [],
                  "FN": []}
    val_data = {"Loss": [], "Acc": [], "F1": [], "Dice": [], "Pre": [], "Re": [], "TP": [], "TN": [], "FP": [],
                "FN": []}
    test_data = {"Loss": [], "Acc": [], "F1": [], "Dice": [], "Pre": [], "Re": [], "TP": [], "TN": [], "FP": [],
                 "FN": []}
    data_dict = {
        "[TRAIN]": train_data,
        "[EVAL]": val_data,
        "[TEST]": test_data
    }

    # parse data
    with open(log_file, 'r') as f:
        lines = f.read()
    lines = lines.split('\n')
    for line in lines:
        # split into metrics
        fields = line.split(',')
        log_mode = fields[0].split(' ')[0]
        if log_mode != "":
            for field in fields:
                field = field.split(':')
                metric = field[-2].strip()
                metric_value = float(field[-1].strip())
                # check keys before append new data to dictionary
                assert log_mode in data_dict.keys()
                assert metric in data_dict[log_mode].keys()
                data_dict[log_mode][metric].append(metric_value)

    # validate data reading
    if len(data_dict["[TRAIN]"]) != len(data_dict["[EVAL]"]):
        print("[ERROR] Train size and validation size mismatch!")

    return data_dict


def __plot_fig__(fig_name, x, ys, y_names, bound_value, save_dir=None, no_show=False):
    # init figure
    fig = plt.figure(figsize=(10, 10))
    plt.title(fig_name)
    plt.xlabel("epochs")

    # draw
    for i, y_name in enumerate(y_names):
        y = ys[i]
        # min max scatter
        y_max_index = np.argmax(y)
        y_min_index = np.argmin(y)
        plt.text(x[y_max_index], y[y_max_index], f"{y_max_index}, {y[y_max_index]:.2f}", size='small')
        plt.text(x[y_min_index], y[y_min_index], f"{y_min_index}, {y[y_min_index]:.2f}", size='small')
        plt.scatter([x[y_max_index], x[y_min_index]], [y[y_max_index], y[y_min_index]], s=25)
        # line
        plt.plot(x, y, label=y_name)
    margin = [bound_value] * len(x)
    plt.plot(x, margin)
    plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, fig_name + ".png"))
    if not no_show:
        plt.show()


def __show_fig__(data_dict, save_dir=None, is_track_early_stop=False, no_show=False):
    x = range(len(data_dict["[TRAIN]"]["Loss"]))  # same for the other metrics

    # plot loss train & eval
    __plot_fig__(fig_name="train-val_loss", x=x, ys=[data_dict["[TRAIN]"]["Loss"], data_dict["[EVAL]"]["Loss"]],
                 y_names=["train loss", "validation loss"], bound_value=0, save_dir=save_dir, no_show=no_show)

    # plot accuracy train & eval
    __plot_fig__(fig_name="train-val_accuracy-dice", x=x,
                 ys=[data_dict["[TRAIN]"]["Acc"], data_dict["[EVAL]"]["Acc"], data_dict["[TRAIN]"]["Dice"],
                     data_dict["[EVAL]"]["Dice"]],
                 y_names=["train accuracy", "validation accuracy", "train dice", "validation dice"], bound_value=100,
                 save_dir=save_dir, no_show=no_show)

    # plot train
    __plot_fig__(fig_name="train-metrics", x=x,
                 ys=[data_dict["[TRAIN]"]["Acc"], data_dict["[TRAIN]"]["F1"], data_dict["[TRAIN]"]["Dice"]],
                 y_names=["train accuracy", "train f1", "train dice"], bound_value=100, save_dir=save_dir,
                 no_show=no_show)

    # plot eval
    __plot_fig__(fig_name="validation-metrics", x=x,
                 ys=[data_dict["[EVAL]"]["Acc"], data_dict["[EVAL]"]["F1"], data_dict["[EVAL]"]["Dice"]],
                 y_names=["validation accuracy", "validation f1", "validation dice"], bound_value=100,
                 save_dir=save_dir, no_show=no_show)

    if not is_track_early_stop:
        # plot test
        __plot_fig__(fig_name="test-metrics", x=x,
                     ys=[data_dict["[TEST]"]["Acc"], data_dict["[TEST]"]["F1"], data_dict["[TEST]"]["Dice"]],
                     y_names=["test accuracy", "test f1", "test dice"], bound_value=100, save_dir=save_dir,
                     no_show=no_show)


def __cal_stats__(data_dict, metrics, start_i, end_i, save_dir):
    with open(os.path.join(save_dir, "test_stats.txt"), 'w') as f:
        for metric in metrics:
            # calculate mean, standard deviation and standard error
            log_mode, metric_name = metric.split(':')
            metric_scores = data_dict[log_mode][metric_name][start_i:end_i]
            mean = round(np.mean(metric_scores), 6)
            std = round(np.std(metric_scores), 6)
            std_err = round(std / math.sqrt(len(metric_scores)), 6)
            # write to file
            f.write('\n')
            f.write(f"{metric}:\n")
            f.write(f"Mean: {mean}, Standard Deviation: {std}, Standard Error: {std_err}\n")


# process the record to plot figures and calculate statistics
def process_log(log_file, save_dir, is_track_early_stop=False, no_show=False):
    data_dict = __parse_data__(log_file)
    __show_fig__(data_dict, save_dir=save_dir, is_track_early_stop=is_track_early_stop, no_show=no_show)
    if not is_track_early_stop:
        __cal_stats__(
            data_dict=data_dict,
            metrics=["[EVAL]:Acc", "[EVAL]:F1", "[EVAL]:Dice", "[TEST]:Acc", "[TEST]:F1", "[TEST]:Dice"],
            start_i=149,
            end_i=-1,
            save_dir=save_dir
        )


if __name__ == '__main__':
    log_file = "../covid_ncpz_checkpoints/covid_log.txt"
    save_dir = os.path.join(os.path.dirname(log_file), "stats")
    process_log(log_file, save_dir=save_dir, is_track_early_stop=True, no_show=False)
