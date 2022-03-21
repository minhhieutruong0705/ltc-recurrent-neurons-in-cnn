import numpy as np
import torch
from tqdm import tqdm


class DiabeticRetinopathyTrainer():
    def __init__(self, model, train_loader, loss_function, optimizer, scaler, device):
        self.model = model
        self.loader = train_loader
        self.loss_fn = loss_function
        self.optim = optimizer
        self.scaler = scaler
        self.device = device

    def train(self):
        total_train_loss = 0
        tp = tn = fp = fn = 0

        loop = tqdm(self.loader)
        self.model.train()

        for batch_index, (image, label) in enumerate(loop):
            x = image.to(self.device)
            label = label.long().to(self.device)

            with torch.cuda.amp.autocast():
                prediction = self.model(x)
                ground_truth = torch.zeros_like(prediction, device=self.device)
                ground_truth[np.arange(x.size(0)), label] = 1
                loss = self.loss_fn(prediction, ground_truth)
                loss_no_grad = loss

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            # pause gradient calculation
            with torch.no_grad():
                prediction_no_grad = prediction.argmax(dim=1, keepdim=True)
                ground_truth_no_grad = label.view_as(prediction_no_grad)

                tp += (prediction_no_grad * ground_truth_no_grad).sum()
                tn += ((1 - prediction_no_grad).abs() * (1 - ground_truth_no_grad).abs()).sum()
                fp += (prediction_no_grad * (1 - ground_truth_no_grad).abs()).sum()
                fn += ((1 - prediction_no_grad).abs() * ground_truth_no_grad).sum()

            loop.set_postfix(loss=loss_no_grad.item())
            total_train_loss += loss_no_grad.item()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = 2 * precision * recall / (precision + recall) * 100
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        dice = tp / (fp + fn + tp) * 100
        avg_train_loss = total_train_loss / len(self.loader)

        print("\n[TRAIN]:          Training Loss: {:.6f}".format(avg_train_loss))
        print(f"[Classification]: Dice: {dice:2f}, Acc: {accuracy:2f}, F1: {f1:2f},")
        print(f"[Confusion]:      TP: {tp.item()}, TN: {tn.item()}, FP: {fp.item()}, FN: {fn.item()}")
        return avg_train_loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn
