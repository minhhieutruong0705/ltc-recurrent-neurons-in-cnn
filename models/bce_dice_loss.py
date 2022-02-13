import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLossWithLogistic(nn.Module):
    def __init__(self, reduction="mean", bce_weight=0.4, dice_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, ground_truth):
        assert predictions.size() == ground_truth.size()
        assert ground_truth.max() <= 1 and ground_truth.min() >= 0

        # dice loss
        predictions_sigmoid = torch.sigmoid(predictions)
        intersection = (predictions_sigmoid * ground_truth).sum()
        union = predictions_sigmoid.sum() + ground_truth.sum()
        dice_loss = 1 - (2 * intersection) / union

        # bce loss
        bce_loss = F.binary_cross_entropy_with_logits(
            input=predictions,
            target=ground_truth,
            reduction=self.reduction
        )

        bce_dice_loss = bce_loss * self.bce_weight + dice_loss * self.dice_weight
        return bce_dice_loss


if __name__ == '__main__':
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16, 2)).float()
    loss_fn = BCEDiceLossWithLogistic()
    loss = loss_fn(x, y)
    print(loss)