import torch
import torch.nn as nn


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
def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("[INFO] Checkpoint loaded")
