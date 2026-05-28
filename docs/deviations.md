# Deviations

Intentional departures from a literal reading of the manuscript, each with its paper anchor and
justification. Anything not listed here is implemented as written.

## D1. Conformal target coverage default

- Anchor: Methods Eq. 11 and Fig. 1 caption state alpha = 0.05; Results (Selective deferral analysis)
  report coverage "above the specified 90% threshold", which implies alpha = 0.10.
- Decision: the default `SplitConformal.alpha = 0.10`, matching the reported empirical coverage. The
  0.05 figure value is exposed as an override.
- Impact: changes only the nominal coverage level; the procedure is unchanged.

## D2. Frozen-backbone substrate

- Anchor: Methods, MLLM backbone (LLaVA-Med 7B frozen) and Implementation details (A100 80GB).
- Decision: the default backbone is a deterministic analytic feature map (DCT spectral vision features +
  character n-gram text hashing through a frozen seeded-orthonormal projection). The real
  `LlavaMedBackbone` / `BiomedClipBackbone` adapters load actual weights when `transformers`/`open_clip`
  and the weights are present, and raise at the I/O boundary otherwise.
- Justification: the manuscript's claim is that the framework is backbone-agnostic (the uncertainty,
  fusion, adaptation, and deferral modules are identical across LLaVA-Med and BiomedCLIP). The analytic
  substrate exercises that exact scientific core on CPU without distributing 7B weights, while the real
  adapters preserve the full pipeline for an A100 run. The trainable adaptation pathway (low-rank
  residual standing in for LoRA), visual projection, SNGP, Dirichlet, fusion, and prototype temperature
  are real modules optimised in every run.
- Impact: numerical results require the real backbone and the licensed datasets; the analytic substrate
  is for development, tests, and the smoke configuration only.

## D3. MRI center-slice count

- Anchor: Methods, MLLM backbone: "we extract the three center slices from each orientation (sagittal,
  coronal, axial)".
- Decision: `TriOrientationPool.slices_per_orientation = 3` (nine slices total), configurable.
- Impact: pooling input count only; the mean-pool operation is unchanged.

## D4. Training precision

- Anchor: Implementation details name PyTorch 2.1.0 on A100 80GB but do not state fp16/bf16/tf32.
- Decision: default `precision = "bf16"` (A100-appropriate); fp16 and fp32 are selectable.
- Impact: numerical precision of the forward/backward pass only.
