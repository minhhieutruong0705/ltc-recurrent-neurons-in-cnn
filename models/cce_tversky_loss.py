import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCCEFocalTverskyLossWithSoftmax(nn.Module):
    def __init__(self, class_weights, reduction="mean",
                 cce_weight=0.1, dice_weight=1.0,
                 tversky_alpha=0.7, focal_gamma=0.75, device="cuda"):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
        self.cce_weight = cce_weight
        self.dice_weight = dice_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = 1 - tversky_alpha
        self.focal_gamma = focal_gamma
        self.device = device

    def forward(self, predictions, ground_truth):
        assert predictions.size() == ground_truth.size()
        assert ground_truth.max() <= 1 and ground_truth.min() >= 0

        # activation
        predictions_softmax = predictions.softmax(dim=1)

        prediction_category = predictions_softmax.argmax(dim=1)
        ground_truth_category = ground_truth.argmax(dim=1)

        # # micro dice loss
        # similarity_mask = torch.eq(prediction_category, ground_truth_category)
        # intersection = (ground_truth_category.bool() * similarity_mask).sum()
        # union = prediction_category.bool().sum() + ground_truth_category.bool().sum()
        # dice_loss = 1 - (2 * intersection) / union

        # confusion matrix
        n_classes = len(self.class_weights)
        confusion_matrix = torch.zeros((n_classes, n_classes), device=self.device)
        for truth, prediction in zip(ground_truth_category, prediction_category):
            confusion_matrix[truth, prediction] += 1
        tp_classes = confusion_matrix.diag()
        fp_classes = confusion_matrix.sum(dim=0) - tp_classes
        fn_classes = confusion_matrix.sum(dim=1) - tp_classes

        # weighted dice loss
        esp = 1e-8
        tversky = (tp_classes + esp) / \
                  (tp_classes + fp_classes * self.tversky_beta + fn_classes * self.tversky_alpha + esp)
        focal_tversky_loss = torch.pow((1 - tversky), self.focal_gamma)
        weighted_focal_tversky_loss = (focal_tversky_loss * self.class_weights).sum() / \
                                      self.class_weights[ground_truth_category].sum()

        # cce loss
        weighted_cce_loss = F.cross_entropy(
            input=predictions_softmax,
            target=ground_truth,
            reduction=self.reduction,
            weight=self.class_weights
        )

        cce_dice_loss = weighted_cce_loss * self.cce_weight + weighted_focal_tversky_loss * self.dice_weight
        return cce_dice_loss


if __name__ == '__main__':
    x = torch.randn(16, 5)
    y = torch.randint(0, 1, (16, 5)).float()
    weight = torch.randn(5)
    loss_fn = WeightedCCEFocalTverskyLossWithSoftmax(class_weights=weight, device="cpu")
    loss = loss_fn(x, y)
    print(loss)
