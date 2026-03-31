"""
Gaia Transformer Model Architecture.

Multi-layer Transformer Decoder for soil microbiome modeling.
Designed for continual pre-training from MGM weights.

Architecture:
  - Input: [genus1_token] [genus2_token] ... [genus512_token]
           (sorted by abundance, descending)
  - Model: Multi-layer Transformer Decoder
  - Pre-training task: Causal Language Modeling (next token prediction)
  - Additional tasks: Masked reconstruction, abundance prediction
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GaiaConfig:
    """Configuration for Gaia Transformer model."""

    def __init__(
        self,
        vocab_size: int = 5000,
        max_length: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_token_id = pad_token_id

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GaiaConfig":
        return cls(**d)


class GaiaTransformer(nn.Module):
    """
    Gaia Foundation Model — Transformer Decoder for microbiome sequences.

    Supports:
    - Causal LM (next token prediction)
    - Masked token reconstruction
    - Embedding extraction for downstream tasks
    """

    def __init__(self, config: GaiaConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.position_encoding = PositionalEncoding(
            config.d_model, config.max_length, config.dropout
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.n_layers
        )

        # Output head (language model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights: embedding <-> lm_head
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask (upper triangular)."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def _generate_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate padding mask from input IDs."""
        return input_ids == self.config.pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            labels: (batch_size, seq_len) target token IDs for LM loss
            return_embeddings: If True, return hidden states

        Returns:
            Dictionary with 'logits', optionally 'loss' and 'embeddings'
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)

        # Masks
        causal_mask = self._generate_causal_mask(seq_len, input_ids.device)
        padding_mask = self._generate_padding_mask(input_ids)

        # Decoder (self-attention only, no cross-attention)
        # Use decoder with memory = zeros (decoder-only architecture)
        memory = torch.zeros(
            batch_size, 1, self.config.d_model, device=input_ids.device
        )
        x = self.decoder(
            x,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # LM head
        logits = self.lm_head(x)

        output = {"logits": logits}

        if return_embeddings:
            output["embeddings"] = x

        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            output["loss"] = loss

        return output

    def get_sample_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get a fixed-size embedding for each sample by mean-pooling
        non-padding hidden states.

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            (batch_size, d_model) sample embeddings
        """
        with torch.no_grad():
            output = self.forward(input_ids, return_embeddings=True)
            embeddings = output["embeddings"]

            # Mean pool over non-padding positions
            mask = (input_ids != self.config.pad_token_id).unsqueeze(-1).float()
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return pooled

    def get_attention_weights(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract attention weights from all layers for interpretability.

        Returns:
            List of (batch_size, n_heads, seq_len, seq_len) tensors
        """
        # Hook to capture attention weights
        attention_weights = []

        def hook_fn(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1])

        hooks = []
        for layer in self.decoder.layers:
            h = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(h)

        # Need to temporarily enable attention weight output
        for layer in self.decoder.layers:
            layer.self_attn.need_weights = True

        with torch.no_grad():
            self.forward(input_ids)

        # Cleanup
        for h in hooks:
            h.remove()
        for layer in self.decoder.layers:
            layer.self_attn.need_weights = False

        return attention_weights
