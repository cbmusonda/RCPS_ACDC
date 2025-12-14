import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler


class BaseSegmentationModel(nn.Module):
    """
    Lightweight base class for segmentation models.

    NOTE:
    - This version is **not** an abstract base class.
    - It removes all @abstractmethod decorators so that subclasses
      like SemiSupervisedContrastiveSegmentationModel can be instantiated
      normally in single-GPU Colab.
    - Subclasses are expected to override:
        * set_input(...)
        * set_test_input(...)
        * optimize_parameters(...)
        * evaluate_one_step(...)
    """

    def __init__(self, cfg: Dict[str, Any], num_classes: int, amp: bool = False) -> None:
        super().__init__()

        self.cfg = cfg
        self.num_classes = num_classes
        self.amp_enabled = amp

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # AMP scaler (warning about deprecation is harmless)
        self.scaler = GradScaler(enabled=self.amp_enabled)

        # training state
        self.start_epoch: int = 0
        self.best_metric: float = -1.0
        self.best_metric_epoch: int = -1

        # placeholders that subclasses or training script may fill
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.metric_meter: Optional[Any] = None
        self.train_loss: Optional[Any] = None
        self.val_loss: Optional[Any] = None

    # ------------------------------------------------------------------
    # Methods expected to be overridden by subclasses
    # ------------------------------------------------------------------
    def set_input(self, *args, **kwargs) -> None:
        """
        Set training inputs (labeled + unlabeled batches).
        Subclasses should override.
        """
        raise NotImplementedError(
            "set_input must be implemented in the subclass."
        )

    def set_test_input(self, *args, **kwargs) -> None:
        """
        Set validation / test inputs.
        Subclasses should override.
        """
        raise NotImplementedError(
            "set_test_input must be implemented in the subclass."
        )

    def optimize_parameters(self, *args, **kwargs) -> None:
        """
        One optimization step over a batch.
        Subclasses should override.
        """
        raise NotImplementedError(
            "optimize_parameters must be implemented in the subclass."
        )

    def evaluate_one_step(self, *args, **kwargs) -> None:
        """
        One evaluation step over a batch (e.g. compute metrics, save preds).
        Subclasses should override.
        """
        raise NotImplementedError(
            "evaluate_one_step must be implemented in the subclass."
        )

    # ------------------------------------------------------------------
    # Checkpointing utilities
    # ------------------------------------------------------------------
    def _get_network_state(self) -> Dict[str, Any]:
        """
        Handle both plain nn.Module and DDP-wrapped modules.
        """
        net = getattr(self, "network", None)
        if net is None:
            return {}
        if hasattr(net, "module"):  # DDP
            return net.module.state_dict()
        return net.state_dict()

    def save_networks(self, epoch: int, save_dir: str, filename: str = "latest.pt") -> None:
        """
        Save network + optimizer + scheduler + basic training state.

        Args:
            epoch: current epoch
            save_dir: directory to save checkpoints
            filename: checkpoint filename (default: latest.pt)
        """
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, filename)

        state = {
            "epoch": epoch,
            "network": self._get_network_state(),
            "best_metric": getattr(self, "best_metric", None),
            "best_metric_epoch": getattr(self, "best_metric_epoch", None),
        }

        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()

        torch.save(state, ckpt_path)
        print(f"[Checkpoint] Saved to: {ckpt_path}")

    def load_networks(self, ckpt_path: str, resume_training: bool = True) -> None:
        """
        Load network (and optionally optimizer/scheduler and training state).

        Args:
            ckpt_path: path to .pt checkpoint
            resume_training: if True, also load optimizer, scheduler, epoch, metrics
        """
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"[Checkpoint] Loading from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # load network weights
        net = getattr(self, "network", None)
        if net is None:
            raise RuntimeError("BaseSegmentationModel has no attribute 'network' to load weights into.")

        if hasattr(net, "module"):  # DDP case
            net.module.load_state_dict(ckpt["network"])
        else:
            net.load_state_dict(ckpt["network"])

        if resume_training:
            if "optimizer" in ckpt and self.optimizer is not None:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt["scheduler"])

            self.start_epoch = int(ckpt.get("epoch", 0))
            self.best_metric = float(ckpt.get("best_metric", -1.0))
            self.best_metric_epoch = int(ckpt.get("best_metric_epoch", -1))
            print(
                f"[Checkpoint] Resumed from epoch {self.start_epoch}, "
                f"best metric {self.best_metric} at epoch {self.best_metric_epoch}"
            )

    # ------------------------------------------------------------------
    # Logging hooks (no-op by default; WandBModel may extend)
    # ------------------------------------------------------------------
    def log_train_loss(self, step: int) -> None:
        """
        Hook called in the training loop to log training losses.
        In this base class it's a no-op; WandBModel or subclasses can override.
        """
        # Example of what a subclass could do:
        # if hasattr(self, "train_loss") and hasattr(self, "wandb_run"):
        #     self.wandb_run.log(self.train_loss.to_dict(step), step=step)
        pass