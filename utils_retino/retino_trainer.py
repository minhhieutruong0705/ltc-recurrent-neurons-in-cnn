import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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
        total_prediction_category = []
        total_ground_truth_category = []

        loop = tqdm(self.loader)
        self.model.train()

        for index, (image, label) in enumerate(loop):
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
                prediction_category = prediction.argmax(dim=1)
                ground_truth_category = label.view_as(prediction_category)

                # store prediction and ground truth of each iteration
                total_prediction_category.extend(prediction_category.detach().cpu().numpy())
                total_ground_truth_category.extend(ground_truth_category.detach().cpu().numpy())

            loop.set_postfix(loss=loss_no_grad.item())
            total_train_loss += loss_no_grad.item()

        assert len(total_prediction_category) == len(total_ground_truth_category)

        # calculate accuracy
        accuracy = accuracy_score(total_ground_truth_category, total_prediction_category) * 100

        # calculate precision, recall, anf f1 score (macro)
        precision = precision_score(total_ground_truth_category, total_prediction_category, average="macro")
        recall = recall_score(total_ground_truth_category, total_prediction_category, average="macro")
        f1 = f1_score(total_ground_truth_category, total_prediction_category, average="macro") * 100

        # confusion matrix for each class
        matrix = confusion_matrix(total_ground_truth_category, total_prediction_category)
        fp_classes = matrix.sum(axis=0) - np.diag(matrix)
        fn_classes = matrix.sum(axis=1) - np.diag(matrix)
        tp_classes = np.diag(matrix)
        tn_classes = matrix.sum() - (fp_classes + fn_classes + tp_classes)

        # macro dice for positive classes
        dice_classes = tp_classes / (tp_classes + fp_classes + fn_classes)
        dice = np.average(dice_classes[1:]) * 100

        # confusion considering class 0 is negative and others are positive
        tp = tp_classes[1:].sum()
        tn = tp_classes[0]
        fp = fp_classes[1:].sum()
        fn = fp_classes[0]

        avg_train_loss = total_train_loss / len(self.loader)

        print("\n[TRAIN]:          Training Loss: {:.6f}".format(avg_train_loss))
        print(f"[Classification]: Dice: {dice:2f}, Acc: {accuracy:2f}, F1: {f1:2f},")
        print(f"[Confusion]:      TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")  # all classes
        return avg_train_loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn
