# Implementation Map

The source tree carries no comments or docstrings; this table is the single record tying each paper
item to the code that realises it. Columns: paper location, equation/figure/table, file path, symbol,
notes.

## Equations

| paper location | item | file | symbol | notes |
|---|---|---|---|---|
| Methods, Problem formulation | Eq. 1 (composite objective) | `src/uamfst/fining/adjustments.py` | `CompositeObjective` | `L = L_CE + lambda_cal * L_ECE + lambda_dir * KL[Dir(alpha)||Dir(1)]`; lambda_cal=0.1, lambda_dir=0.01 |
| Methods, Problem formulation | Eq. 1 (ECE term) | `src/uamfst/fining/clarity.py` | `differentiable_ece` | soft-binned differentiable expected calibration error |
| Methods, MLLM backbone | Eq. 2 (visual projection) | `src/uamfst/pressing/extraction.py` | `VisualProjection` | `h = v W_proj + b`, 1024 -> 4096 (768 -> 4096 for BiomedCLIP variant) |
| Methods, Uncertainty decomposition | Eq. 3 (epistemic variance) | `src/uamfst/maceration/tannins.py` | `mc_dropout_variance`, `combine_epistemic` | MC-Dropout sample variance over T passes, combined with SNGP closed-form variance |
| Methods, Uncertainty decomposition | Eq. 3 (SNGP closed form) | `src/uamfst/fermentation/yeast.py` | `RandomFeatureGP` | 1,024 random Fourier features; spectral-normalised two-layer head 4096->512->256 |
| Methods, Uncertainty decomposition | Eq. 4 (Dirichlet head) | `src/uamfst/fermentation/malolactic.py` | `DirichletHead` | `alpha = g_phi(h) + 1`, g_phi = 256->128->K |
| Methods, Uncertainty decomposition | Eq. 5 (aleatoric entropy) | `src/uamfst/fermentation/malolactic.py` | `expected_categorical_entropy` | entropy of the mean categorical of the Dirichlet |
| Methods, Fusion gate | Eq. 6 (fused representation) | `src/uamfst/maceration/marrying.py` | `UncertaintyFusionGate` | `z = (w_v h_v + w_t h_t)/(w_v + w_t)` |
| Methods, Fusion gate | Eq. 7 (precision weights) | `src/uamfst/maceration/marrying.py` | `precision_weight` | `w_m = 1/(sigma2_epi_m + eps)`, eps=1e-6 |
| Methods, Few-shot adapter | Eq. 8 (uncertainty-weighted prototype) | `src/uamfst/grafting/scions.py` | `build_prototypes` | `p_c = mean_i z_i/(sigma2_epi(x_i)+eps)` over support set |
| Methods, Few-shot adapter | Eq. 9 (nearest-prototype softmax) | `src/uamfst/grafting/scions.py` | `PrototypeClassifier` | `p(y=c|x) ~ exp(-||z - p_c||^2 / (2 tau^2))`, learnable tau init 0.07 |
| Methods, Conformal deferral | Eq. 10 (nonconformity) | `src/uamfst/appellation/controlled_origin.py` | `nonconformity_scores` | `s = 1 - p(y|x) + lambda_unc (sigma2_epi + sigma2_alea)`, lambda_unc=0.5 |
| Methods, Conformal deferral | Eq. 11 (prediction set) | `src/uamfst/appellation/controlled_origin.py` | `SplitConformal` | quantile `q_hat_{ceil((1-alpha)(n+1))/n}`; finite-sample coverage |
| Methods, Conformal deferral | dual deferral criterion | `src/uamfst/appellation/declassification.py` | `DualCriterionDeferral` | defer if `|C(x)| > kappa` (kappa=1) OR `sigma2_epi > theta` (85th pct of calibration epistemic) |

## Architecture and backbone

| paper location | item | file | symbol | notes |
|---|---|---|---|---|
| Methods, MLLM backbone | LLaVA-Med 7B | `src/uamfst/pressing/varietals.py` | `LlavaMedBackbone` | lazy `transformers`/`peft`; raises at I/O boundary if weights absent |
| Methods, cross-architecture | BiomedCLIP 86M | `src/uamfst/pressing/varietals.py` | `BiomedClipBackbone` | ViT-B/16 + PubMedBERT; d=768 |
| Methods, MLLM backbone | MRI tri-slice pooling | `src/uamfst/pressing/crushing.py` | `TriOrientationPool` | center slices per orientation, encode independently, mean-pool |
| Methods, multimodal text input | structured clinical text | `src/uamfst/pressing/bouquet.py` | `ClinicalTextTemplate` | builds structured text from metadata fields per dataset |
| (deterministic substrate) | analytic backbone | `src/uamfst/pressing/must.py` | `AnalyticBackbone` | DCT spectral vision features + n-gram-hash text; frozen seeded-orthonormal projection |
| Methods, MLLM backbone | LoRA-analogue adapter | `src/uamfst/pressing/fortification.py` | `LowRankAdapter` | trainable low-rank residual on frozen features (r=16) |
| Fig. 3 / Fig. 4 | full estimator | `src/uamfst/cuverie/cuvee.py` | `UAMFST` | composes backbone -> projection -> SNGP/Dirichlet -> fusion -> head |

## Datasets and pipeline

| paper location | item | file | symbol | notes |
|---|---|---|---|---|
| Datasets, MRNet | MRNet adapter | `src/uamfst/vineyard/mrnet.py` | `MRNetDataset` | official split; sagittal/coronal/axial volumes |
| Datasets, MURA | MURA adapter | `src/uamfst/vineyard/mura.py` | `MuraDataset` | 7 regions; binary normal/abnormal |
| Datasets, OAI | OAI adapter | `src/uamfst/vineyard/oai.py` | `OAIDataset` | subject-level splits; cross-domain transfer only |
| Datasets, VQA | VQA-RAD / SLAKE | `src/uamfst/vineyard/vqa.py` | `VQADataset` | image-level splits |
| Datasets | transforms | `src/uamfst/vineyard/sorting.py` | `build_transform` | 224x224, flip, +/-15 deg rotation, color jitter |
| Data leakage prevention | manifest hashing | `src/uamfst/vineyard/consignment.py` | `manifest_sha256` | SHA-256 over fitted preprocessing manifest |
| (synthetic) | in-memory cohort | `src/uamfst/vineyard/nursery.py` | `SyntheticCohort` | deterministic paired image/text for tests and smoke |

## Training, evaluation, statistics

| paper location | item | file | symbol | notes |
|---|---|---|---|---|
| Implementation details | trainer | `src/uamfst/maturation/aging.py` | `TrainingLoop` | AdamW, cosine+warmup, AMP, grad-accum, EMA |
| Implementation details | optimizer/schedule | `src/uamfst/maturation/racking.py` | `build_optimizer`, `build_schedule` | AdamW lr 2e-5 wd 1e-2; cosine over 50 epochs, 5 warmup |
| R4 | EMA | `src/uamfst/maturation/lees.py` | `ExponentialMovingAverage` | shadow weights |
| R4 | atomic checkpoint + seed | `src/uamfst/maturation/bottling.py` | `save_checkpoint`, `load_checkpoint` | tmp file + os.replace; stores seed |
| R4 | seeding | `src/uamfst/maturation/inoculation.py` | `set_seed` | torch/numpy/python |
| Evaluation metrics | AUC/sens/spec | `src/uamfst/tasting/palate.py` | `roc_auc`, `sensitivity_specificity` | threshold-free AUC |
| Evaluation metrics, ECE | ECE 15-bin | `src/uamfst/tasting/palate.py` | `expected_calibration_error` | equal-width 15 bins |
| Evaluation metrics, Brier | Brier score | `src/uamfst/tasting/palate.py` | `brier_score` | |
| Statistical significance | DeLong test | `src/uamfst/tasting/panel.py` | `delong_test` | correlated-AUC variance |
| Statistical significance | bootstrap CI | `src/uamfst/tasting/panel.py` | `bootstrap_ci` | 1,000 resamples |
| Statistical significance | Holm-Bonferroni | `src/uamfst/tasting/panel.py` | `holm_bonferroni` | 12 pre-specified comparisons |
| Table 2 | interaction ratio | `src/uamfst/tasting/balance.py` | `interaction_ratio` | combined drop / sum of individual drops |
| selective prediction | coverage / retained acc | `src/uamfst/tasting/flight.py` | `coverage`, `retained_accuracy` | deferral operating points |
| Results / Eval loops | evaluator | `src/uamfst/tasting/judging.py` | `Evaluator` | assembles the per-task report |

## Tables, figures, ablations -> configs

| paper item | config |
|---|---|
| Table 1 main result | `configs/experiment/main.gin` |
| Table 1 BiomedCLIP row / Suppl. S8 | `configs/experiment/supplementary_biomedclip.gin` |
| Table 2 Tier 1 (single-component removal) | `configs/experiment/ablation_no_uq.gin`, `ablation_no_fusion.gin`, `ablation_no_adapter.gin` |
| Table 2 Tier 2 (pairwise removal) | `configs/experiment/ablation_uq_fusion.gin`, `ablation_uq_adapter.gin`, `ablation_fusion_adapter.gin` |
| Table 2 Tier 2 (base MLLM) | `configs/experiment/ablation_base.gin` |
| Table 2 Tier 3 (temperature scaling) | `configs/experiment/ablation_temperature.gin` |
| Table 3 few-shot (1/2/4/8/16-shot) | `configs/experiment/main.gin` with `UAMFST.shots` override |
| Suppl. S9 medical VQA | `configs/experiment/supplementary_vqa.gin` |
| MURA primary | `configs/experiment/main_mura.gin` |
| OAI cross-domain transfer | `configs/experiment/supplementary_oai_transfer.gin` |
| pytest smoke | `configs/experiment/_smoke.gin` |

## Figures

| figure | realised by |
|---|---|
| Fig. 1 micro-anatomy / ambiguity | conceptual; no code |
| Fig. 2 uncertainty heterogeneity + calibration | `tasting/flight.py` + epistemic-prevalence correlation in `tasting/panel.py` |
| Fig. 3 framework overview | `cuverie/cuvee.py` |
| Fig. 4 architecture | `cuverie/cuvee.py` + `fermentation/*` + `maceration/*` |
