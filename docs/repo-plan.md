# Repository Plan

## Layout

```
uamfst/
  README.md  LICENSE  pyproject.toml  requirements.txt  environment.yml
  Dockerfile  Makefile  .gitignore  .pre-commit-config.yaml
  configs/
    model/        backbone + head gin fragments (llava_med, biomedclip, heads)
    data/         dataset gin fragments (mrnet, mura, oai, vqa, synthetic)
    train/        optimization + schedule gin fragments
    experiment/   main, ablation_*, supplementary_*, _smoke
  src/uamfst/
    charter/        covenants (typing.Protocol) + vessels (frozen dataclass records)
    vineyard/       harvest, parcel, consignment, sorting, baskets, nursery, mrnet, mura, oai, vqa
    pressing/       varietals, must (analytic substrate), extraction (Eq 2), crushing, bouquet, fortification
    fermentation/   yeast (SNGP, Eq 3), malolactic (Dirichlet, Eqs 4-5)
    maceration/     punchdown (mc-dropout), tannins (epistemic combiner, Eq 3), marrying (fusion gate, Eqs 6-7)
    grafting/       scions (few-shot prototypes, Eqs 8-9)
    fining/         clarity (ece), adjustments (composite objective, Eq 1), cold_stabilization, stability
    appellation/    controlled_origin (conformal, Eqs 10-11), declassification (dual deferral)
    cuverie/        manifold (typed gin shim), cuvee (full UAMFST), assemblage (factory registration)
    maturation/     aging (training loop), racking (optimization), lees (ema), bottling (checkpoint), inoculation (seeding)
    tasting/        palate (metrics), panel (statistics), balance (interaction), flight (selective), judging (evaluate)
    cooperage/      logbook (logging), crew (distributed), casks (io)
    cellarmaster/   absl entrypoints: vintage (train), scoring (evaluate), pour (infer), label (export), station (runtime)
  scripts/        launch_train.sh, launch_eval.sh, prepare_data.sh
  tests/          varied unit + smoke suites
  docs/           project-context, implementation-map, deviations
  assets/
```

## Module responsibilities

- **charter** — `VisionLanguageBackbone`, `UncertaintyModule`, `DeferralPolicy` protocols; `BackboneOutput`,
  `UncertaintyReadout`, `PredictionBundle`, `DatasetSpec`, `RunConfig` dataclasses. No torch logic.
- **pressing** — turn raw image/text into modality hidden states; the analytic substrate keeps the
  package runnable without 7B weights.
- **fermentation** — the two probabilistic objects: a random-feature Gaussian process (SNGP) and a
  Dirichlet output layer with entropy readout.
- **maceration** — turn the distributions into epistemic/aleatoric scalars and fuse modalities by
  precision.
- **grafting** — few-shot prototype construction and classification.
- **fining** — the calibration loss term and the composite training objective; a post-hoc
  temperature-scaling baseline for the Tier-3 ablation.
- **appellation** — split-conformal prediction sets and the dual-criterion deferral router.
- **cuverie** — the typed gin shim and the composed estimator; the only place gin and torch meet.
- **vineyard / maturation / tasting / cooperage / cellarmaster** — data, training, evaluation,
  infrastructure, CLI.

## Dependencies (pinned in requirements.txt)

torch==2.1.0, numpy>=1.24, scipy>=1.10, gin-config>=0.5.0, absl-py>=1.4, scikit-learn>=1.3,
pillow>=10.0. Dev: pytest, ruff, black, isort, mypy, pre-commit. Optional (real backbones, not imported
by default): transformers==4.36.0, peft, open_clip_torch.

## Test coverage targets

- fermentation: SNGP feature shape + variance positivity; Dirichlet entropy monotonicity.
- maceration: precision-weight monotonicity; fusion convexity; epistemic combiner.
- grafting: prototype recovery; nearest-prototype separability.
- fining: ECE against synthetic calibrated/miscalibrated logits.
- appellation: empirical conformal coverage >= 1 - alpha on exchangeable data; deferral routing.
- tasting: AUC vs scikit-learn; DeLong sanity; bootstrap-CI width; interaction ratio.
- maturation: overfit single batch; gradient flow (trainable vs frozen); seed determinism; checkpoint
  round-trip.
- end-to-end: `tests/test_training_smoke.py` 2 steps on `_smoke.gin`, loss decreases.
- hygiene: no forbidden phrases / emoji / source comments.
