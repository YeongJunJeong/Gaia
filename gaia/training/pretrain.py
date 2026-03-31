"""
Pre-training Script for Gaia.

Continual pre-training from MGM weights on the soil microbiome corpus.

Usage:
    python -m gaia.training.pretrain \
        --corpus data/processed/gaia-corpus-v1.pkl \
        --vocab data/processed/tokenizer.json \
        --output checkpoints/pretrain \
        [--mgm-weights path/to/mgm_weights.pt]
"""

import argparse
import json
import logging

import torch

from gaia.models.transformer import GaiaConfig, GaiaTransformer
from gaia.training.dataset import MicrobiomeDataset
from gaia.training.trainer import GaiaTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_mgm_weights(model: GaiaTransformer, weights_path: str) -> GaiaTransformer:
    """
    Load MGM pre-trained weights for continual pre-training.

    Handles potential architecture mismatches by loading only
    compatible parameters.
    """
    logger.info(f"Loading MGM weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")

    # Handle wrapped state dicts
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Load compatible parameters
    model_dict = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    model_dict.update(compatible)
    model.load_state_dict(model_dict)

    logger.info(
        f"Loaded {len(compatible)}/{len(state_dict)} parameters "
        f"(skipped {len(skipped)} incompatible)"
    )
    if skipped:
        logger.info(f"Skipped parameters: {skipped[:10]}...")

    return model


def main():
    parser = argparse.ArgumentParser(description="Gaia pre-training")
    parser.add_argument("--corpus", required=True, help="Path to corpus pickle")
    parser.add_argument("--vocab", required=True, help="Path to tokenizer JSON")
    parser.add_argument("--output", default="checkpoints/pretrain")
    parser.add_argument("--mgm-weights", default=None, help="MGM weights for transfer")

    # Model config
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=2048)

    # Training config
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    # Load vocab size from tokenizer
    with open(args.vocab) as f:
        tokenizer_data = json.load(f)
    vocab_size = len(tokenizer_data["vocab"])
    max_length = tokenizer_data["max_length"]

    # Create model
    config = GaiaConfig(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
    )
    model = GaiaTransformer(config)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Load MGM weights if available
    if args.mgm_weights:
        model = load_mgm_weights(model, args.mgm_weights)

    # Dataset
    dataset = MicrobiomeDataset(
        corpus_path=args.corpus,
        mask_ratio=args.mask_ratio,
    )
    logger.info(f"Dataset: {len(dataset)} samples")

    # Training
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        checkpoint_dir=args.output,
        use_wandb=args.wandb,
        wandb_run_name="gaia-pretrain",
    )

    trainer = GaiaTrainer(model, dataset, training_config)
    history = trainer.train()

    # Save final model
    final_path = f"{args.output}/gaia-v0.1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "history": history,
        },
        final_path,
    )
    logger.info(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
