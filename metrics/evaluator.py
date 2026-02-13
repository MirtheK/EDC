import logging
from collections.abc import Sequence
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import ModuleList
from torchmetrics import Metric


logger = logging.getLogger(__name__)


class Evaluator(nn.Module, Callback):
    """Evaluator module for LightningModule.

    The Evaluator module is a PyTorch module that computes and logs metrics during
    validation and test steps. Each AnomalibModule should have an Evaluator module as
    a submodule to compute and log metrics during validation and test steps. An Evaluation
    module can be passed to the AnomalibModule as a parameter during initialization. When
    no Evaluator module is provided, the AnomalibModule will use a default Evaluator module
    that logs a default set of metrics.

    Args:
        val_metrics (Sequence[Metric], optional): Validation metrics.
            Defaults to ``[]``.
        test_metrics (Sequence[Metric], optional): Test metrics.
            Defaults to ``[]``.
        compute_on_cpu (bool, optional): Whether to compute metrics on CPU.
            Defaults to ``True``.
    """

    def __init__(
        self,
        val_metrics: Metric | Sequence[Metric] | None = None,
        test_metrics: Metric | Sequence[Metric] | None = None,
        compute_on_cpu: bool = True,
    ) -> None:
        super().__init__()
        self.val_metrics = ModuleList(self.validate_metrics(val_metrics))
        self.test_metrics = ModuleList(self.validate_metrics(test_metrics))
        self.compute_on_cpu = compute_on_cpu

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Move metrics to cpu if ``num_devices == 1`` and ``compute_on_cpu`` is set to ``True``."""
        del pl_module, stage  # Unused arguments.
        if trainer.num_devices > 1:
            if self.compute_on_cpu:
                logger.warning("Number of devices is greater than 1, setting compute_on_cpu to False.")
        elif self.compute_on_cpu:
            self.metrics_to_cpu(self.val_metrics)
            self.metrics_to_cpu(self.test_metrics)

    @staticmethod
    def validate_metrics(metrics: Metric | Sequence[Metric] | None) -> Sequence[Metric]:
        """Validate metrics."""
        if metrics is None:
            return []
        if isinstance(metrics, Metric):
            return [metrics]
        if not isinstance(metrics, Sequence):
            msg = f"metrics must be an Metric or a list of Metrics, got {type(metrics)}"
            raise TypeError(msg)
        return metrics

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update validation metrics with the batch output."""
        del trainer, outputs, batch_idx, dataloader_idx, pl_module  # Unused arguments.
        for metric in self.val_metrics:
            metric.update(batch)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Compute and log validation metrics."""
        del pl_module  # Unused argument.
        for metric in self.val_metrics:
            self.log(metric.name, metric)
            # In barebones mode, logging is disabled. We manually update trainer metrics
            # to ensure they're available in both callback_metrics and the validate() return value
            if trainer.barebones:
                metric_value = metric.compute()
                trainer.callback_metrics[metric.name] = metric_value
                trainer.logged_metrics[metric.name] = metric_value

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update test metrics with the batch output."""
        del trainer, outputs, batch_idx, dataloader_idx, pl_module  # Unused arguments.
        for metric in self.test_metrics:
            metric.update(batch)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Compute and log test metrics."""
        del pl_module  # Unused argument.
        for metric in self.test_metrics:
            self.log(metric.name, metric)
            # In barebones mode, logging is disabled. We manually update trainer metrics
            # to ensure they're available in both callback_metrics and the return value of the trainer.test() method
            if trainer.barebones:
                metric_value = metric.compute()
                trainer.callback_metrics[metric.name] = metric_value
                trainer.logged_metrics[metric.name] = metric_value

    def metrics_to_cpu(self, metrics: Metric | list[Metric] | ModuleList) -> None:
        """Set the compute_on_cpu attribute of the metrics to True."""
        if isinstance(metrics, Metric):
            metrics.compute_on_cpu = True
        elif isinstance(metrics, (list | ModuleList)):
            for metric in metrics:
                self.metrics_to_cpu(metric)
        else:
            msg = f"metrics must be a Metric or a list of metrics, got {type(metrics)}"
            raise TypeError(msg)