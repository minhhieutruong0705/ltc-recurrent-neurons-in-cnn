import os
import numpy as np
import math

from utils_plot import plot_fig


def track_training(data_dict, training_name, **plt_kwarg):
    n_epochs = len(data_dict["[TRAIN]"]["Loss"])
    x = range(n_epochs)  # scatter x based on number of epochs
    print(f"[INFO] Training of {training_name} is at {n_epochs} epochs!")

    # plot train loss
    plot_fig(fig_name=f"{training_name}_train_loss", x=x,
             ys=[data_dict["[TRAIN]"]["Loss"]], y_names=["train loss"], bound_value=0, **plt_kwarg)

    # plot validation loss
    plot_fig(fig_name=f"{training_name}_validation_loss", x=x,
             ys=[data_dict["[VALID]"]["Loss"]], y_names=["validation loss"], bound_value=0, **plt_kwarg)

    # plot train & eval accuracy & dice
    plot_fig(fig_name=f"{training_name}_train-val_metrics", x=x,
             ys=[data_dict["[TRAIN]"]["Acc"], data_dict["[VALID]"]["Acc"],
                 data_dict["[TRAIN]"]["F1"], data_dict["[VALID]"]["F1"],
                 data_dict["[TRAIN]"]["Dice"], data_dict["[VALID]"]["Dice"]],
             y_names=["train accuracy", "validation accuracy",
                      "train f1", "validation f1",
                      "train dice", "validation dice"], bound_value=100,
             **plt_kwarg)

    # plot train
    plot_fig(fig_name=f"{training_name}_train-metrics", x=x,
             ys=[data_dict["[TRAIN]"]["Acc"], data_dict["[TRAIN]"]["F1"], data_dict["[TRAIN]"]["Dice"]],
             y_names=["train accuracy", "train f1", "train dice"], bound_value=100, **plt_kwarg)

    # plot eval
    plot_fig(fig_name=f"{training_name}_validation-metrics", x=x,
             ys=[data_dict["[VALID]"]["Acc"], data_dict["[VALID]"]["F1"], data_dict["[VALID]"]["Dice"]],
             y_names=["validation accuracy", "validation f1", "validation dice"], bound_value=100, **plt_kwarg)


def get_stats(data_dict, metrics, start_i, end_i, save_file_path):
    with open(save_file_path, 'w') as f:
        for metric in metrics:
            # get scores
            log_mode, metric_name = metric.split(':')  # parse input
            metric_scores = data_dict[log_mode][metric_name][start_i:end_i]
            assert len(metric_scores) > 0

            # calculate mean, standard deviation, standard error, max, min
            mean = np.mean(metric_scores)
            std = np.std(metric_scores)
            std_err = std / math.sqrt(len(metric_scores))
            max_score = np.max(metric_scores)
            min_score = np.min(metric_scores)

            # write to file
            f.write('\n')
            f.write(f"{metric}:\n")
            f.write(f"Mean: {mean:.6f}, Standard Deviation: {std:.6f}, Standard Error: {std_err:.6f}\n"
                    f"Max: {max_score:.2f}, Min: {min_score:.2f}\n")
