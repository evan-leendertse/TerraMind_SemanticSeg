import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: (N, C, H, W) after sigmoid/softmax
        targets: (N, C, H, W) one-hot encoded or same shape
        """
        # flatten
        preds = preds.contiguous().view(preds.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)

        intersection = (preds * targets).sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return 1 - dice_score.mean()