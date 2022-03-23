import os
import numpy as np
import torch
import torch.nn as nn
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


# call back by a model to initialize its weights
def init_weights(module):
    if type(module) == nn.Linear or \
            type(module) == nn.Conv2d or \
            type(module) == nn.ConvTranspose2d:
        torch.nn.init.kaiming_normal_(module.weight)
    elif type(module) == nn.BatchNorm2d:
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# record process
def log_to_file(file, mode, epoch, loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn):
    with open(file, 'a+') as f:
        f.write(f"\n[{mode}] Epoch {epoch}: "
                f"Loss: {loss:.4f}, "
                f"Acc: {accuracy:.2f}, "
                f"F1: {f1:.2f}, "
                f"Dice: {dice:.2f}, "
                f"Pre: {precision:.2f}, "
                f"Re: {recall:.2f}, "
                f"TP: {tp}, "
                f"TN: {tn}, "
                f"FP: {fp}, "
                f"FN: {fn}")


# save checkpoint
def save_checkpoint(state, checkpoint_file, checkpoint_index=None):
    torch.save(state, checkpoint_file)
    if checkpoint_index is not None:
        torch.save(state, checkpoint_file.replace(".pth.tar", str(checkpoint_index) + ".pth.tar"))
    print("[INFO] Checkpoint saved!")


# load checkpoint
def load_checkpoint(checkpoint_file, model, optimizer):
    checkpoint_state = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint_state["state_dict"])
    optimizer.load_state_dict(checkpoint_state["optimizer"])
    print("[INFO] Checkpoint loaded")
    return checkpoint_state["epoch"]


# draw confusion matrix
def draw_confusion_matrix(matrix, class_names, fig_name, save_dir, normalize=True):
    if normalize:
        matrix = matrix.astype(float) / matrix.sum(axis=1)[:, np.newaxis]
    # convert to dataframe
    matrix_df = pd.DataFrame(
        data=matrix,
        index=[class_name for class_name in class_names],
        columns=[class_name for class_name in class_names]
    )

    # plot figure
    plt.figure(figsize=(12, 7))
    plt.title(fig_name)
    heat_map = sn.heatmap(matrix_df, annot=True)
    heat_map.set(xlabel="Prediction", ylabel="Ground Truth")
    plt.savefig(os.path.join(save_dir, f"{fig_name}.png"))
    plt.close()
