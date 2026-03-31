"""Tests for preprocessing modules."""

import numpy as np
import pandas as pd
import pytest


def _make_abundance_df(n_samples=50, n_genera=100):
    """Create a synthetic abundance DataFrame for testing."""
    np.random.seed(42)
    data = np.random.exponential(scale=10, size=(n_samples, n_genera))
    # Add some zeros (sparsity)
    mask = np.random.random((n_samples, n_genera)) < 0.3
    data[mask] = 0

    genus_names = [f"Genus_{i}" for i in range(n_genera)]
    df = pd.DataFrame(data, columns=genus_names)
    df.insert(0, "sample_id", [f"sample_{i}" for i in range(n_samples)])
    return df


def _make_metadata_df(n_samples=50):
    """Create a synthetic metadata DataFrame for testing."""
    biomes = ["agricultural soil", "forest soil", "grassland soil", "desert", ""]
    return pd.DataFrame(
        {
            "sample_id": [f"sample_{i}" for i in range(n_samples)],
            "biome": [biomes[i % len(biomes)] for i in range(n_samples)],
            "latitude": np.random.uniform(-90, 90, n_samples),
            "longitude": np.random.uniform(-180, 180, n_samples),
            "sequencing_platform": ["Illumina"] * (n_samples - 5) + [None] * 5,
        }
    )


class TestTSSNormalization:
    def test_sums_to_one(self):
        from gaia.preprocessing.normalization import tss_normalize

        df = _make_abundance_df()
        result = tss_normalize(df)
        genus_cols = [c for c in result.columns if c != "sample_id"]
        row_sums = result[genus_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_preserves_sample_ids(self):
        from gaia.preprocessing.normalization import tss_normalize

        df = _make_abundance_df()
        result = tss_normalize(df)
        assert list(result["sample_id"]) == list(df["sample_id"])


class TestCLRNormalization:
    def test_output_shape(self):
        from gaia.preprocessing.normalization import clr_normalize

        df = _make_abundance_df()
        result = clr_normalize(df)
        assert result.shape == df.shape

    def test_centered(self):
        from gaia.preprocessing.normalization import clr_normalize

        df = _make_abundance_df()
        result = clr_normalize(df)
        genus_cols = [c for c in result.columns if c != "sample_id"]
        row_means = result[genus_cols].mean(axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-10)


class TestSparsityFiltering:
    def test_removes_rare_genera(self):
        from gaia.preprocessing.filtering import filter_sparse_genera

        df = _make_abundance_df()
        # Set one genus to be present in only 1 sample
        genus_cols = [c for c in df.columns if c != "sample_id"]
        df[genus_cols[0]] = 0
        df.iloc[0, 1] = 100  # Only present in 1 sample

        result = filter_sparse_genera(df, min_prevalence=0.05)
        assert genus_cols[0] not in result.columns

    def test_preserves_common_genera(self):
        from gaia.preprocessing.filtering import filter_sparse_genera

        df = _make_abundance_df()
        result = filter_sparse_genera(df, min_prevalence=0.001)
        assert result.shape[1] > 1  # At least sample_id + some genera


class TestMetadataStandardization:
    def test_envo_mapping(self):
        from gaia.preprocessing.metadata import standardize_biome

        assert standardize_biome("agricultural soil") == "ENVO:00002259"
        assert standardize_biome("forest") == "ENVO:01001198"
        assert standardize_biome("unknown biome type") == "unknown"
        assert standardize_biome(None) == "unknown"

    def test_standardize_metadata(self):
        from gaia.preprocessing.metadata import standardize_metadata

        df = _make_metadata_df()
        result = standardize_metadata(df)
        assert "biome_envo" in result.columns
        assert "metadata_complete" in result.columns
        assert "batch_correctable" in result.columns


class TestTokenizer:
    def test_build_vocab(self):
        from gaia.preprocessing.tokenizer import MicrobiomeTokenizer

        df = _make_abundance_df()
        tokenizer = MicrobiomeTokenizer(max_length=64)
        vocab = tokenizer.build_vocab(df)
        assert len(vocab) > 5  # At least special tokens + some genera
        assert "[PAD]" in vocab
        assert "[MASK]" in vocab

    def test_encode_decode(self):
        from gaia.preprocessing.tokenizer import MicrobiomeTokenizer

        df = _make_abundance_df(n_samples=5, n_genera=20)
        tokenizer = MicrobiomeTokenizer(max_length=32)
        tokenizer.build_vocab(df)

        genus_cols = [c for c in df.columns if c != "sample_id"]
        encoded = tokenizer.encode(df.iloc[0], genus_cols)
        assert len(encoded) == 32
        assert encoded[0] == tokenizer.vocab["[CLS]"]

    def test_batch_encode(self):
        from gaia.preprocessing.tokenizer import MicrobiomeTokenizer

        df = _make_abundance_df(n_samples=10, n_genera=20)
        tokenizer = MicrobiomeTokenizer(max_length=32)
        tokenizer.build_vocab(df)

        batch = tokenizer.encode_batch(df)
        assert batch.shape == (10, 32)
