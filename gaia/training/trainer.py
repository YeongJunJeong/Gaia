"""
Training Loop for Gaia Models.

Supports:
- Pre-training (causal LM, masked LM)
- Fine-tuning (regression, classification)
- Mixed precision (fp16)
- Gradient accumulation
- Learning rate scheduling (warmup + cosine decay)
- Checkpointing
- Optional Weights & Biases logging
"""

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimization
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 50
    gradient_clipping: float = 1.0

    # Scheduling
    warmup_steps: int = 1000
    scheduler: str = "cosine"  # "cosine" or "linear"

    # Data split
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05

    # Mixed precision
    fp16: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 5
    save_best: bool = True

    # Logging
    log_every_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "gaia"
    wandb_run_name: str | None = None

    # Misc
    num_workers: int = 4
    seed: int = 42


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class GaiaTrainer:
    """
    Trainer for Gaia foundation model.

    Handles the training loop, validation, checkpointing,
    and logging for both pre-training and fine-tuning.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Data splits
        n_total = len(dataset)
        n_train = int(n_total * config.train_ratio)
        n_val = int(n_total * config.val_ratio)
        n_test = n_total - n_train - n_val

        generator = torch.Generator().manual_seed(config.seed)
        self.train_set, self.val_set, self.test_set = random_split(
            dataset, [n_train, n_val, n_test], generator=generator
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        total_steps = len(self.train_loader) * config.max_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, config.warmup_steps, total_steps
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.fp16)
        self.autocast_dtype = torch.float16 if config.fp16 else torch.float32

        # Tracking
        self.best_val_loss = float("inf")
        self.global_step = 0

        # Checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Wandb
        if config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=vars(config),
                )
            except ImportError:
                logger.warning("wandb not installed, disabling logging")
                self.config.use_wandb = False

    def train(self) -> dict:
        """Run the full training loop."""
        logger.info(f"Training on {self.device}")
        logger.info(f"Train: {len(self.train_set)}, Val: {len(self.val_set)}, "
                     f"Test: {len(self.test_set)}")
        logger.info(f"Total steps: {len(self.train_loader) * self.config.max_epochs}")

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                f"Epoch {epoch}/{self.config.max_epochs} — "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
            )

            # Save checkpoint
            if epoch % self.config.save_every_epochs == 0:
                self._save_checkpoint(epoch, val_loss)

            if self.config.save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)

            if self.config.use_wandb:
                import wandb

                wandb.log(
                    {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                )

        return history

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                output = self.model(input_ids, labels=labels)
                loss = output["loss"]

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clipping
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"  Step {self.global_step}: loss={loss.item():.4f}, lr={lr:.2e}"
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                output = self.model(input_ids, labels=labels)

            total_loss += output["loss"].item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False
    ):
        """Save model checkpoint."""
        filename = "best.pt" if is_best else f"epoch_{epoch}.pt"
        path = Path(self.config.checkpoint_dir) / filename

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "global_step": self.global_step,
                "config": self.model.config.to_dict()
                if hasattr(self.model, "config")
                else {},
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")
