from __future__ import annotations

import torch
from torch import nn

from uamfst.charter.covenants import VisionLanguageBackbone
from uamfst.charter.vessels import Batch, PredictionBundle, UncertaintyReadout
from uamfst.fermentation.malolactic import DirichletHead, expected_categorical_entropy
from uamfst.fermentation.yeast import RandomFeatureGP
from uamfst.grafting.scions import PrototypeClassifier, build_prototypes
from uamfst.maceration.marrying import UncertaintyFusionGate
from uamfst.maceration.punchdown import stochastic_passes
from uamfst.maceration.tannins import EpistemicCombiner, mc_dropout_variance
from uamfst.pressing.crushing import TriOrientationPool
from uamfst.pressing.extraction import VisualProjection
from uamfst.pressing.fortification import LowRankAdapter


class UAMFST(nn.Module):
    def __init__(
        self,
        backbone: VisionLanguageBackbone,
        num_classes: int = 2,
        sngp_hidden: int = 512,
        sngp_proj: int = 256,
        sngp_features: int = 1024,
        dirichlet_hidden: int = 128,
        adapter_rank: int = 16,
        dropout: float = 0.1,
        mc_passes: int = 10,
        use_uncertainty: bool = True,
        use_fusion: bool = True,
        use_adapter: bool = True,
    ) -> None:
        super().__init__()
        self.pool = TriOrientationPool(backbone)
        self.model_dim = self.pool.text_dim
        self.num_classes = num_classes
        self.mc_passes = mc_passes
        self.use_uncertainty = use_uncertainty
        self.use_fusion = use_fusion
        self.use_adapter = use_adapter
        self.projection = VisualProjection(self.pool.vision_dim, self.model_dim)
        self.adapter = LowRankAdapter(self.model_dim, rank=adapter_rank)
        self.fusion = UncertaintyFusionGate()
        self.stream_dropout = nn.Dropout(dropout)
        self.fused_dropout = nn.Dropout(dropout)
        self.sngp = RandomFeatureGP(
            input_dim=self.model_dim,
            hidden_dim=sngp_hidden,
            proj_dim=sngp_proj,
            num_features=sngp_features,
            num_classes=num_classes,
        )
        self.dirichlet = DirichletHead(sngp_proj, dirichlet_hidden, num_classes)
        self.combiner = EpistemicCombiner()
        self.prototype = PrototypeClassifier()
        for parameter in self.pool.backbone.parameters():
            parameter.requires_grad_(False)

    def encode(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        vision = self.projection(self.pool.encode_vision(batch.images))
        text = self.pool.encode_text(batch.token_ids)
        return vision, text

    def _stream_predictive(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.sngp.projection(self.stream_dropout(hidden))
        features = self.sngp.random_features(projected)
        return torch.softmax(self.sngp.logits(features), dim=-1)

    def _modality_epistemic(self, hidden: torch.Tensor, passes: int) -> torch.Tensor:
        samples = stochastic_passes(lambda: self._stream_predictive(hidden), self, passes)
        return mc_dropout_variance(samples)

    def _fused_predictive(self, fused: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.sngp(self.fused_dropout(fused))
        return torch.softmax(logits, dim=-1)

    def _zero(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(reference.shape[0])

    def _apply_adapter(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.use_adapter:
            return hidden
        adapted: torch.Tensor = self.adapter(hidden)
        return adapted

    def forward(self, batch: Batch, passes: int | None = None) -> PredictionBundle:
        steps = passes if passes is not None else self.mc_passes
        vision, text = self.encode(batch)
        if self.use_uncertainty:
            epistemic_vision = self._modality_epistemic(vision, steps)
            epistemic_text = self._modality_epistemic(text, steps)
        else:
            epistemic_vision = self._zero(vision)
            epistemic_text = self._zero(vision)
        if self.use_fusion:
            fused = self.fusion(vision, text, epistemic_vision, epistemic_text)
        else:
            fused = vision
        fused = self._apply_adapter(fused)
        logits, projected, features = self.sngp(self.fused_dropout(fused))
        concentration = self.dirichlet(projected)
        aleatoric = expected_categorical_entropy(concentration)
        if self.use_uncertainty:
            sngp_variance = self.sngp.variance(features.detach())
            dropout_variance = mc_dropout_variance(
                stochastic_passes(lambda: self._fused_predictive(fused.detach()), self, steps)
            )
            epistemic = self.combiner(sngp_variance, dropout_variance)
        else:
            epistemic = self._zero(vision)
        readout = UncertaintyReadout(
            epistemic=epistemic,
            aleatoric=aleatoric,
            epistemic_vision=epistemic_vision,
            epistemic_text=epistemic_text,
        )
        return PredictionBundle(
            logits=logits,
            probabilities=torch.softmax(logits, dim=-1),
            concentration=concentration,
            fused=fused,
            uncertainty=readout,
        )

    def fused_features(self, batch: Batch, passes: int | None = None) -> torch.Tensor:
        steps = passes if passes is not None else self.mc_passes
        vision, text = self.encode(batch)
        if not self.use_fusion:
            return self._apply_adapter(vision)
        if self.use_uncertainty:
            epistemic_vision = self._modality_epistemic(vision, steps)
            epistemic_text = self._modality_epistemic(text, steps)
        else:
            epistemic_vision = self._zero(vision)
            epistemic_text = self._zero(vision)
        fused: torch.Tensor = self.fusion(vision, text, epistemic_vision, epistemic_text)
        return self._apply_adapter(fused)

    def adapt_prototypes(self, support: Batch, passes: int | None = None) -> torch.Tensor:
        features = self.fused_features(support, passes)
        bundle = self.forward(support, passes)
        return build_prototypes(
            features,
            bundle.uncertainty.epistemic,
            support.labels,
            self.num_classes,
        )

    def refresh_precision(self, fused: torch.Tensor) -> None:
        _, _, features = self.sngp(fused)
        self.sngp.accumulate_precision(features.detach())
