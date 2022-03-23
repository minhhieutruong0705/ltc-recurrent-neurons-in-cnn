import torch
import torch.nn as nn
import torch.nn.functional as F


class CCEMicroDiceLossWithSoftmax(nn.Module):
    def __init__(self, reduction="mean", cce_weight=1.0, dice_weight=0.3, class_weights=None):
        super().__init__()
        self.reduction = reduction
        self.cce_weight = cce_weight
        self.dice_weight = dice_weight
        self.class_weights = class_weights

    def forward(self, predictions, ground_truth):
        assert predictions.size() == ground_truth.size()
        assert ground_truth.max() <= 1 and ground_truth.min() >= 0

        # micro dice loss
        prediction_binary = predictions.argmax(dim=1, keepdim=True) > 0
        ground_truth_binary = ground_truth.argmax(dim=1, keepdim=True) > 0
        intersection = (prediction_binary * ground_truth_binary).sum()
        union = prediction_binary.sum() + ground_truth_binary.sum()
        dice_loss = 1 - (2 * intersection) / union

        # activation
        predictions_softmax = predictions.softmax(dim=1)

        # cce loss
        cce_loss = F.cross_entropy(
            input=predictions_softmax,
            target=ground_truth,
            reduction=self.reduction,
            weight=self.class_weights
        )

        cce_dice_loss = cce_loss * self.cce_weight + dice_loss * self.dice_weight
        return cce_dice_loss


if __name__ == '__main__':
    x = torch.randn(16, 5)
    y = torch.randint(0, 2, (16, 5)).float()
    loss_fn = CCEMicroDiceLossWithSoftmax()
    loss = loss_fn(x, y)
    print(loss)
