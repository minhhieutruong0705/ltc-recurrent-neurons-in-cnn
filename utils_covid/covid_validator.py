import numpy as np
import torch
from tqdm import tqdm


class CovidValidator():
    def __init__(self, model, val_loader, loss_function, device, is_test=False):
        self.model = model
        self.loader = val_loader
        self.loss_fn = loss_function
        self.device = device
        self.is_test = is_test

    def eval(self):
        total_eval_loss = 0
        tp = tn = fp = fn = 0

        loop = tqdm(self.loader)
        self.model.eval()

        with torch.no_grad():
            for batch_index, (image, label) in enumerate(loop):
                x = image.to(self.device)
                label = label.to(self.device)
                prediction = self.model(x)

                prediction_no_grad = prediction.argmax(dim=1, keepdim=True)
                ground_truth_no_grad = label.view_as(prediction_no_grad)
                loss = self.loss_fn(prediction_no_grad.float(), ground_truth_no_grad.float())

                tp += (prediction_no_grad * ground_truth_no_grad).sum()
                tn += ((1 - prediction_no_grad).abs() * (1 - ground_truth_no_grad).abs()).sum()
                fp += (prediction_no_grad * (1 - ground_truth_no_grad).abs()).sum()
                fn += ((1 - prediction_no_grad).abs() * ground_truth_no_grad).sum()

                loop.set_postfix(loss=loss.item())
                total_eval_loss += loss.item()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = 2 * precision * recall / (precision + recall) * 100
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        dice = tp / (fp + fn + tp) * 100
        avg_eval_loss = total_eval_loss / len(self.loader)

        print(f"\n[{'VALID' if not self.is_test else 'TEST'}]:          "
              f"{'Validation' if not self.is_test else 'Test'} Loss: {avg_eval_loss:.6f}")
        print(f"[Classification]: Dice: {dice:2f}, Acc: {accuracy:2f}, F1: {f1:2f},")
        print(f"[Confusion]:      TP: {tp.item()}, TN: {tn.item()}, FP: {fp.item()}, FN: {fn.item()}")
        return avg_eval_loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn
