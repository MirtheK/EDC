import torch
from matplotlib.figure import Figure
from torchmetrics.classification.roc import BinaryROC
from torchmetrics.utilities.compute import auc



class AUROC(BinaryROC):
    """Area under the ROC curve.

    This class computes the area under the receiver operating characteristic
    curve, which plots the true positive rate against the false positive rate
    at various thresholds.

    Examples:
        To compute the metric for a set of predictions and ground truth targets:
        >>> from anomalib.metrics.auroc import _AUROC
        >>> import torch
        >>> preds = torch.tensor([0.13, 0.26, 0.08, 0.92, 0.03])
        >>> target = torch.tensor([0, 0, 1, 1, 0])
        >>> auroc = _AUROC()
        >>> auroc(preds, target)
        tensor(0.6667)

        It is possible to update the metric state incrementally:

        >>> auroc.update(preds[:2], target[:2])
        >>> auroc.update(preds[2:], target[2:])
        >>> auroc.compute()
        tensor(0.6667)

        To plot the ROC curve:

        >>> figure, title = auroc.generate_figure()
    """

    def compute(self) -> torch.Tensor:
        """First compute ROC curve, then compute area under the curve.

        Returns:
            torch.Tensor: Value of the AUROC metric
        """
        tpr: torch.Tensor
        fpr: torch.Tensor

        fpr, tpr = self._compute()
        return auc(fpr, tpr, reorder=True)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new predictions and targets.

        Need to flatten new values as ROC expects them in this format for binary
        classification.

        Args:
            preds (torch.Tensor): Predictions from the model
            target (torch.Tensor): Ground truth target labels
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute false positive rate and true positive rate value pairs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing tensors for FPR
                and TPR values
        """
        tpr: torch.Tensor
        fpr: torch.Tensor
        fpr, tpr, _thresholds = super().compute()
        return (fpr, tpr)

