"""Test computing the accuracy metric."""

import torch

from mantis.metrics.accuracy import Accuracy


def test_accuracy_range() -> None:
    """Test the accuracy metric."""
    metric = Accuracy(topk=5)
    num_classes = 10
    batch_size = 16
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    metric(logits, labels)
    acc = metric.compute()
    assert 0 <= acc <= 1


def test_compare_top1_topk() -> None:
    """Test that acc top1 is always smaller than top5."""
    metric_top1 = Accuracy(topk=1)
    metric_top5 = Accuracy(topk=5)
    batch_size = 16
    num_classes = 10
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    metric_top1(logits, labels)
    metric_top5(logits, labels)
    acc_top1 = metric_top1.compute()
    acc_top5 = metric_top5.compute()
    assert acc_top1 <= acc_top5
