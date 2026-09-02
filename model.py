"""Neural modules for the stable two-view ST-MCC implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_DIM = 128
COMMON_DIM = 64


class MLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.extend((nn.LayerNorm(dims[index + 1]), nn.GELU()))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GradientReverse(torch.autograd.Function):
    """Identity in the forward pass, sign reversal in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


@dataclass
class ModelOutput:
    latents: list[torch.Tensor]
    commons: list[torch.Tensor]
    specifics: list[torch.Tensor]
    reconstructions: list[torch.Tensor]


class CausalMVC(nn.Module):
    """Two-view common/specific autoencoder with shared global prototypes.

    Cross-view predictors operate only on the common representation.  A
    view-specific latent is intentionally not imputed because information that
    is unique to one view is not identifiable from the other view.
    """

    def __init__(self, input_dims: list[int], num_clusters: int = 5):
        super().__init__()
        if len(input_dims) != 2:
            raise ValueError("The stable model expects exactly two views")

        specific_dim = LATENT_DIM - COMMON_DIM
        self.num_clusters = num_clusters
        self.encoders = nn.ModuleList(
            (
                MLP([input_dims[0], 512, 256, LATENT_DIM]),
                MLP([input_dims[1], 256, 256, LATENT_DIM]),
            )
        )
        self.common_encoder = MLP([LATENT_DIM, LATENT_DIM, COMMON_DIM])
        self.specific_encoders = nn.ModuleList(
            [
                MLP([LATENT_DIM, LATENT_DIM, specific_dim]),
                MLP([LATENT_DIM, LATENT_DIM, specific_dim]),
            ]
        )
        self.decoders = nn.ModuleList(
            (
                MLP([LATENT_DIM, 256, 512, input_dims[0]]),
                MLP([LATENT_DIM, 256, 256, input_dims[1]]),
            )
        )
        self.view_discriminator = MLP([COMMON_DIM, 64, 2])
        # The verified recipe initialized an unused cluster head at this point.
        # Consume the same random draws so existing seed results remain comparable,
        # while keeping the inactive head out of the actual model and checkpoint.
        _legacy_initialization_only = nn.Linear(COMMON_DIM, num_clusters)
        del _legacy_initialization_only
        initial_prototypes = F.normalize(torch.randn(num_clusters, COMMON_DIM), dim=1)
        self.register_buffer("prototypes", initial_prototypes)
        self.register_buffer("imputation_confidence", torch.zeros(2))
        self.cross_predictors = nn.ModuleList(
            (
                MLP([COMMON_DIM, COMMON_DIM, COMMON_DIM]),  # view 0 -> view 1
                MLP([COMMON_DIM, COMMON_DIM, COMMON_DIM]),  # view 1 -> view 0
            )
        )

    def split_latent(self, latent: torch.Tensor, view: int) -> tuple[torch.Tensor, torch.Tensor]:
        common = self.common_encoder(latent)
        specific = self.specific_encoders[view](latent)
        return common, specific

    def decode_latent(self, latent: torch.Tensor, view: int) -> torch.Tensor:
        common, specific = self.split_latent(latent, view)
        return self.decoders[view](torch.cat((common, specific), dim=1))

    def forward(self, views: list[torch.Tensor]) -> ModelOutput:
        latents: list[torch.Tensor] = []
        commons: list[torch.Tensor] = []
        specifics: list[torch.Tensor] = []
        reconstructions: list[torch.Tensor] = []
        for view in range(2):
            latent = self.encoders[view](views[view])
            common, specific = self.split_latent(latent, view)
            reconstruction = self.decoders[view](torch.cat((common, specific), dim=1))
            latents.append(latent)
            commons.append(common)
            specifics.append(specific)
            reconstructions.append(reconstruction)
        return ModelOutput(latents, commons, specifics, reconstructions)

    @torch.no_grad()
    def complete_commons(
        self, views: list[torch.Tensor], mask: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Predict only the missing common representation in both directions."""

        output = self(views)
        completed = [output.commons[0].clone(), output.commons[1].clone()]
        missing_view0 = ~mask[:, 0]
        missing_view1 = ~mask[:, 1]
        completed[0][missing_view0] = self.cross_predictors[1](
            output.commons[1][missing_view0]
        )
        completed[1][missing_view1] = self.cross_predictors[0](
            output.commons[0][missing_view1]
        )
        return completed, output.commons

    @torch.no_grad()
    def calibrate_imputation_confidence(
        self, views: list[torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Estimate direction-wise reliability on genuinely paired samples."""

        output = self(views)
        paired = mask.all(dim=1)
        confidence = output.commons[0].new_zeros(2)
        if paired.any():
            predicted1 = self.cross_predictors[0](output.commons[0][paired])
            predicted0 = self.cross_predictors[1](output.commons[1][paired])
            confidence[0] = F.cosine_similarity(
                predicted0, output.commons[0][paired]
            ).mean().clamp(0.0, 1.0)
            confidence[1] = F.cosine_similarity(
                predicted1, output.commons[1][paired]
            ).mean().clamp(0.0, 1.0)
        self.imputation_confidence.copy_(confidence)
        return confidence.clone()

    @torch.no_grad()
    def completed_common(self, views: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """Return a confidence-weighted completed common representation."""

        completed, _ = self.complete_commons(views, mask)
        observed_weight = torch.ones(len(mask), device=mask.device, dtype=completed[0].dtype)
        weight0 = torch.where(mask[:, 0], observed_weight, self.imputation_confidence[0])
        weight1 = torch.where(mask[:, 1], observed_weight, self.imputation_confidence[1])
        denominator = (weight0 + weight1).clamp_min(1e-8).unsqueeze(1)
        fused = (
            completed[0] * weight0.unsqueeze(1)
            + completed[1] * weight1.unsqueeze(1)
        ) / denominator
        return F.normalize(fused, dim=1)
