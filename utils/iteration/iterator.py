import random
import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler


def set_random_seed(seed: int, benchmark: bool = False):
    """
    Set random seed for python, numpy and torch (both CPU and CUDA),
    and optionally control torch.backends.cudnn.benchmark.

    The 'benchmark' arg is kept for compatibility with train.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Match old behavior: allow enabling/disabling cuDNN benchmark
    torch.backends.cudnn.benchmark = benchmark


class PolynomialLRWithWarmUp(LRScheduler):
    """
    Polynomial learning-rate schedule with optional warm-up.

    Args:
        optimizer: wrapped optimizer.
        total_steps: total number of training steps (usually epochs).
        max_lr_steps: steps over which to apply polynomial decay.
                      If None, use total_steps.
        warmup_steps: number of linear warm-up steps at the beginning.
        last_epoch: index of last epoch (PyTorch convention).
        power: polynomial power.
    
    NOTE: 'verbose' parameter removed for compatibility with PyTorch 2.9+
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        max_lr_steps: int = None,
        warmup_steps: int = 0,
        last_epoch: int = -1,
        power: float = 0.9,
    ):
        self.total_steps = int(total_steps)
        self.max_lr_steps = int(max_lr_steps) if max_lr_steps is not None else int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.power = float(power)

        # PyTorch 2.9+ doesn't accept 'verbose' parameter
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps and self.warmup_steps > 0:
            warmup_factor = float(step + 1) / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        effective_step = min(
            max(step - self.warmup_steps, 0),
            self.max_lr_steps - self.warmup_steps,
        )

        if self.max_lr_steps == self.warmup_steps:
            decay_factor = 1.0
        else:
            progress = float(effective_step) / float(
                self.max_lr_steps - self.warmup_steps
            )
            decay_factor = (1.0 - progress) ** self.power

        return [base_lr * decay_factor for base_lr in self.base_lrs]


class MetricMeter:
    """
    Lightweight metric / loss accumulator.
    """

    def __init__(self, metrics, class_names=None, subject_names=None):
        self.metrics = list(metrics)
        self.class_names = list(class_names) if class_names is not None else []
        self.subject_names = list(subject_names) if subject_names is not None else []
        self.reset()

    def reset(self):
        self.data = {m: [] for m in self.metrics}

    def _add_value(self, metric_name, value):
        try:
            val = float(value)
        except Exception:
            return
        if metric_name in self.data:
            self.data[metric_name].append(val)

    def update(self, metric):
        if isinstance(metric, list):
            for m in metric:
                self.update(m)
            return

        if not isinstance(metric, dict):
            return

        for k, v in metric.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                if len(v) == 0:
                    continue
                value = float(np.mean(v))
            else:
                value = v

            if k in self.data:
                self._add_value(k, value)
                continue

            suffix = k.split("_")[-1]
            if suffix in self.data:
                self._add_value(suffix, value)

    def get_mean(self):
        return {
            m: (float(np.mean(vals)) if len(vals) > 0 else 0.0)
            for m, vals in self.data.items()
        }

    def summary_str(self, prefix: str = "") -> str:
        stats = self.get_mean()
        parts = [f"{m}: {v:.4f}" for m, v in stats.items()]
        if prefix:
            return f"{prefix} " + ", ".join(parts)
        return ", ".join(parts)
    
    def pop_mean_metric(self):
        """Get mean metrics and reset. Used in validation."""
        mean_dict = self.get_mean()
        self.reset()
        return mean_dict