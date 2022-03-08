import os
import numpy as np
import math

from utils_plot import plot_fig


def track_training(data_dict, training_name, include_test=True, **plt_kwarg):
    x = range(len(data_dict["[TRAIN]"]["Loss"]))  # number of epochs

    # plot train & eval loss
    plot_fig(fig_name=f"{training_name}_train-val_loss", x=x,
             ys=[data_dict["[TRAIN]"]["Loss"], data_dict["[EVAL]"]["Loss"]], y_names=["train loss", "validation loss"],
             bound_value=0, **plt_kwarg)

    # plot train & eval accuracy & dice
    plot_fig(fig_name=f"{training_name}_train-val_accuracy-dice", x=x,
             ys=[data_dict["[TRAIN]"]["Acc"], data_dict["[EVAL]"]["Acc"], data_dict["[TRAIN]"]["Dice"],
                 data_dict["[EVAL]"]["Dice"]],
             y_names=["train accuracy", "validation accuracy", "train dice", "validation dice"], bound_value=100,
             **plt_kwarg)

    # plot train
    plot_fig(fig_name=f"{training_name}_train-metrics", x=x,
             ys=[data_dict["[TRAIN]"]["Acc"], data_dict["[TRAIN]"]["F1"], data_dict["[TRAIN]"]["Dice"]],
             y_names=["train accuracy", "train f1", "train dice"], bound_value=100, **plt_kwarg)

    # plot eval
    plot_fig(fig_name=f"{training_name}_validation-metrics", x=x,
             ys=[data_dict["[EVAL]"]["Acc"], data_dict["[EVAL]"]["F1"], data_dict["[EVAL]"]["Dice"]],
             y_names=["validation accuracy", "validation f1", "validation dice"], bound_value=100, **plt_kwarg)

    if include_test:  # after training is finished
        # plot test
        plot_fig(fig_name=f"{training_name}_test-metrics", x=x,
                 ys=[data_dict["[TEST]"]["Acc"], data_dict["[TEST]"]["F1"], data_dict["[TEST]"]["Dice"]],
                 y_names=["test accuracy", "test f1", "test dice"], bound_value=100, **plt_kwarg)


def get_stats(data_dict, metrics, start_i, end_i, save_file_path):
    with open(save_file_path, 'w') as f:
        for metric in metrics:
            # get scores
            log_mode, metric_name = metric.split(':')  # parse input
            metric_scores = data_dict[log_mode][metric_name][start_i:end_i]

            # calculate mean, standard deviation, standard error, max, min
            mean = round(np.mean(metric_scores), 6)
            std = round(np.std(metric_scores), 6)
            std_err = round(std / math.sqrt(len(metric_scores)), 6)
            max_score = np.max(metric_scores)
            min_score = np.min(metric_scores)
            # write to file
            f.write('\n')
            f.write(f"{metric}:\n")
            f.write(f"Mean: {mean:.6f}, Standard Deviation: {std:.6f}, Standard Error: {std_err:.6f}\n"
                    f"Max: {max_score:.2f}, Min: {min_score:.2f}\n")
