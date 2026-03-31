"""
Fine-tuning Script for Gaia.

Supports 4 fine-tuning tasks:
  1. Soil chemistry prediction (regression)
  2. Functional gene abundance prediction
  3. Drought stress classification
  4. Keystone genera identification (interpretability)

Usage:
    python -m gaia.training.finetune \
        --task soil_chemistry \
        --pretrained checkpoints/pretrain/best.pt \
        --corpus data/processed/gaia-corpus-v1.pkl \
        --labels data/processed/gaia-metadata-v1.csv \
        --label-column soilInWaterpH \
        --output checkpoints/finetune/soil_ph
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from gaia.models.transformer import GaiaConfig, GaiaTransformer
from gaia.training.dataset import SupervisedDataset
from gaia.training.trainer import GaiaTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class GaiaForRegression(nn.Module):
    """Gaia with a regression head for continuous value prediction."""

    def __init__(self, backbone: GaiaTransformer, n_outputs: int = 1):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.regression_head = nn.Sequential(
            nn.Linear(backbone.config.d_model, backbone.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(backbone.config.d_model // 2, n_outputs),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # Get sample embedding via mean pooling
        embedding = self.backbone.get_sample_embedding(input_ids)
        predictions = self.regression_head(embedding).squeeze(-1)

        output = {"logits": predictions}

        if labels is not None:
            loss = nn.MSELoss()(predictions, labels.float())
            output["loss"] = loss

        return output


class GaiaForClassification(nn.Module):
    """Gaia with a classification head."""

    def __init__(self, backbone: GaiaTransformer, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.classifier = nn.Sequential(
            nn.Linear(backbone.config.d_model, backbone.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(backbone.config.d_model // 2, n_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        embedding = self.backbone.get_sample_embedding(input_ids)
        logits = self.classifier(embedding)

        output = {"logits": logits}

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels.long())
            output["loss"] = loss

        return output


def load_pretrained_backbone(weights_path: str) -> GaiaTransformer:
    """Load pre-trained Gaia backbone."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    config = GaiaConfig.from_dict(checkpoint["config"])
    model = GaiaTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded pre-trained backbone from {weights_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Gaia fine-tuning")
    parser.add_argument(
        "--task",
        required=True,
        choices=["soil_chemistry", "drought", "tillage", "functional_genes"],
    )
    parser.add_argument("--pretrained", required=True, help="Pre-trained weights")
    parser.add_argument("--corpus", required=True, help="Corpus pickle")
    parser.add_argument("--labels", required=True, help="Labels CSV")
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output", default="checkpoints/finetune")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    # Load backbone
    backbone = load_pretrained_backbone(args.pretrained)

    # Determine task type
    regression_tasks = {"soil_chemistry", "functional_genes"}
    classification_tasks = {"drought", "tillage"}

    if args.task in regression_tasks:
        task_type = "regression"
        model = GaiaForRegression(backbone)
    else:
        task_type = "classification"
        # Need to determine n_classes from data
        import pandas as pd
        labels_df = pd.read_csv(args.labels)
        n_classes = labels_df[args.label_column].dropna().nunique()
        model = GaiaForClassification(backbone, n_classes)
        logger.info(f"Classification task with {n_classes} classes")

    # Freeze backbone if requested
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only fine-tuning head")

    # Dataset
    dataset = SupervisedDataset(
        corpus_path=args.corpus,
        labels_path=args.labels,
        label_column=args.label_column,
        task_type=task_type,
    )
    logger.info(f"Dataset: {len(dataset)} paired samples")

    # Training config (lower LR for fine-tuning)
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        warmup_steps=100,
        checkpoint_dir=args.output,
        use_wandb=args.wandb,
        wandb_run_name=f"gaia-finetune-{args.task}",
    )

    trainer = GaiaTrainer(model, dataset, training_config)
    history = trainer.train()

    # Save final model
    final_path = f"{args.output}/gaia-{args.task}-final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": backbone.config.to_dict(),
            "task": args.task,
            "task_type": task_type,
            "label_column": args.label_column,
            "history": history,
        },
        final_path,
    )
    logger.info(f"Saved fine-tuned model: {final_path}")


if __name__ == "__main__":
    main()
