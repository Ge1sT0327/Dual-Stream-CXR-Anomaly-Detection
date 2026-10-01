"""Context stream: DINOv3 ViT-Large with Multi-Level Feature Aggregation.

Implements the bottom (orange) branch of Figure 1 and Eq. 2 of the paper:

    F_ctx(x) = Concat_{l in S} psi_l(x) = [ psi6(x), psi12(x), psi18(x) ]   (Eq. 2)

with S = {6, 12, 18}, leading to ``F_ctx(x) in R^{N x 3D}`` where N = 196 and
D = 1024. Layer 24 is intentionally excluded: it carries the over-smoothed
"macroscopic" semantics responsible for the semantic gap quantified in
Section 2.2 of the paper (image-level AUC 0.6552 for the vanilla ViT).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import ContextStreamConfig


def assert_patch_grid(
    num_tokens: int, grid: Tuple[int, int] = (14, 14)
) -> Tuple[int, int]:
    """Verify a patch-token count matches the expected ViT-L/16 grid."""
    expected = grid[0] * grid[1]
    if num_tokens != expected:
        raise ValueError(
            f"expected {expected} patch tokens for a {grid[0]}x{grid[1]} grid, "
            f"got {num_tokens}"
        )
    return grid


class ContextStream(nn.Module):
    """Frozen DINOv3 ViT-L/16 encoder exposing intermediate-layer patch tokens."""

    def __init__(self, config: Optional[ContextStreamConfig] = None) -> None:
        super().__init__()
        self.config = config or ContextStreamConfig()
        self.feature_layers: Tuple[int, ...] = tuple(self.config.feature_layers)
        self.vanilla_layer = self.config.vanilla_layer

        self.model = self._build_backbone()
        if self.config.frozen:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

        self.num_register_tokens = int(
            getattr(self.model.config, "num_register_tokens", self.config.num_register_tokens)
        )
        self._hidden_size = int(
            getattr(self.model.config, "hidden_size", self.config.embed_dim)
        )
        self.out_channels: int = self._hidden_size * len(self.feature_layers)

    # ------------------------------------------------------------------ build

    def _build_backbone(self):
        """Load the frozen DINOv3 encoder, never silently hitting the network."""
        from transformers import AutoModel

        kwargs = {"output_hidden_states": True}
        source = self.config.weights_dir or self.config.backbone
        if self.config.weights_dir or self.config.local_files_only:
            # Both branches of the paper's setup expect a local checkpoint; the
            # flag is set in either case so a missing cache fails loudly.
            kwargs["local_files_only"] = True

        return AutoModel.from_pretrained(source, **kwargs)

    # --------------------------------------------------------------- forward

    def patch_tokens(
        self, pixel_values: torch.Tensor, layers: Optional[Sequence[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """Return ``{layer: (B, N, D)}`` patch tokens for the requested layers.

        ``hidden_states`` has length ``num_layers + 1``: index 0 is the embedding
        output, index l is the output of transformer block l. Register tokens and
        the CLS token are stripped so that only the N = 196 patch tokens remain.
        """
        layers = tuple(layers if layers is not None else self.feature_layers)
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:  # pragma: no cover - defensive
            raise RuntimeError("backbone did not return hidden states")

        depth = len(hidden_states) - 1
        for layer in layers:
            if not 1 <= layer <= depth:
                raise ValueError(
                    f"requested layer {layer} but backbone exposes 1..{depth}"
                )

        prefix = 1 + self.num_register_tokens  # CLS + register tokens
        result: Dict[int, torch.Tensor] = {}
        for layer in layers:
            tokens = hidden_states[layer]
            if tokens.shape[0] != pixel_values.shape[0]:
                raise RuntimeError("unexpected batch dimension in hidden states")
            patch = tokens[:, prefix:, :]
            assert_patch_grid(patch.shape[1], self.config.patch_grid)
            result[layer] = patch
        return result

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return ``F_ctx(x)`` with shape ``(B, 196, 3072)`` for a 224x224 input."""
        tokens = self.patch_tokens(pixel_values)
        ordered = [tokens[layer] for layer in self.feature_layers]
        return torch.cat(ordered, dim=-1)  # Eq. 2, channel-wise concatenation

    @torch.no_grad()
    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Inference-mode wrapper used by the pipeline."""
        self.eval()
        return self.forward(pixel_values)

    @torch.no_grad()
    def vanilla_embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Layer-24-only embedding reproducing the Section 2.2 baseline (AUC 0.6552)."""
        return self.patch_tokens(pixel_values, layers=(self.vanilla_layer,))[
            self.vanilla_layer
        ]
