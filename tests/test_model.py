"""Tests for model architecture."""

import torch
import pytest

from gaia.models.transformer import GaiaConfig, GaiaTransformer


@pytest.fixture
def small_config():
    return GaiaConfig(
        vocab_size=100,
        max_length=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    )


@pytest.fixture
def model(small_config):
    return GaiaTransformer(small_config)


class TestGaiaTransformer:
    def test_forward_shape(self, model, small_config):
        batch_size = 4
        input_ids = torch.randint(1, small_config.vocab_size, (batch_size, small_config.max_length))
        output = model(input_ids)
        assert output["logits"].shape == (batch_size, small_config.max_length, small_config.vocab_size)

    def test_loss_computation(self, model, small_config):
        batch_size = 4
        input_ids = torch.randint(1, small_config.vocab_size, (batch_size, small_config.max_length))
        labels = input_ids.clone()
        output = model(input_ids, labels=labels)
        assert "loss" in output
        assert output["loss"].dim() == 0  # Scalar
        assert output["loss"].item() > 0

    def test_embedding_extraction(self, model, small_config):
        batch_size = 4
        input_ids = torch.randint(1, small_config.vocab_size, (batch_size, small_config.max_length))
        embeddings = model.get_sample_embedding(input_ids)
        assert embeddings.shape == (batch_size, small_config.d_model)

    def test_config_serialization(self, small_config):
        d = small_config.to_dict()
        restored = GaiaConfig.from_dict(d)
        assert restored.vocab_size == small_config.vocab_size
        assert restored.d_model == small_config.d_model
        assert restored.n_layers == small_config.n_layers
