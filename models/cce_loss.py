import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCCELoss(nn.Module):
    def __init__(self, class_weights, reduction="mean"):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, predictions, ground_truth):
        assert predictions.size() == ground_truth.size()
        assert ground_truth.max() <= 1 and ground_truth.min() >= 0

        # cce loss
        weighted_cce_loss = F.cross_entropy(
            input=predictions,  # activation included in the criterion
            target=ground_truth,
            reduction=self.reduction,
            weight=self.class_weights
        )

        return weighted_cce_loss


if __name__ == '__main__':
    x = torch.randn(16, 5)
    y = torch.randint(0, 2, (16, 5)).float()
    weight = torch.randn(5)
    loss_fn = WeightedCCELoss(class_weights=weight)
    loss = loss_fn(x, y)
    print(loss)
