# Project Context

Resolved configuration for the UA-MFST code release. Every field records the value we use, the paper
location it was read from, and our confidence.

    project_name       : uamfst                               [HIGH]
    domain             : musculoskeletal / sports-injury dx   [HIGH]
    framework          : PyTorch 2.x + gin-config + absl       [HIGH]
    venue              : npj Digital Medicine                  [MEDIUM-HIGH]
    primary_datasets   : 5 datasets (see §6)                   [HIGH]
    compute_target     : 1x NVIDIA A100 80GB                   [HIGH]
    hparams_reference  : Implementation details p23 + Eqs 1-11 [HIGH]
    supp_path          : none                                  [HIGH]
    extra_signals      : frozen LLaVA-Med-7B + LoRA; empty Code-availability heading; 11 equations

    NEEDS_USER_DECISION: 0 (three low-risk ambiguities defaulted; see §10 and docs/deviations.md)

## 1. project_name

`uamfst` — from the framework name UA-MFST (Uncertainty-Aware Multimodal Few-Shot Transfer).
Source: Title; Abstract L4 ("we present UA-MFST"). Confidence: HIGH.

## 2. supp_path

`none`. Supplementary tables S5, S7, S8, S9 are referenced in the main text (Results, Methods) but no
separate supplementary file ships alongside the manuscript. Confidence: HIGH.

## 3. domain

Musculoskeletal / sports-injury diagnosis with a multimodal medical large language model, uncertainty
quantification (epistemic + aleatoric decomposition), few-shot domain transfer, and conformal selective
deferral for clinical decision support. Source: Title; Abstract; Introduction. Confidence: HIGH.

## 4. framework

PyTorch 2.x with plain `torch.nn` for all modelling. Configuration is expressed with **gin-config**
(`.gin` files) and the command line is driven by **absl** (`absl.app` + `absl.flags`). The paper states
PyTorch 2.1.0 and HuggingFace Transformers 4.36.0 (Implementation details p23); the gin + absl choice is
ours for this release. Confidence: HIGH (torch); chosen (gin + absl).

## 5. venue

npj Digital Medicine. Inferred from the manuscript filename prefix, the structured abstract, and the
Nature-family two-column template with line-numbered methods. Confidence: MEDIUM-HIGH.

## 6. primary_datasets

| name | role | size | license | access |
|---|---|---|---|---|
| MRNet | primary | 1,370 knee MRI exams, 1,250 patients (Stanford, 2001-2012); sagittal T2, coronal T1, axial PD; split 1130/120/120 | Stanford AIMI | https://stanfordmlgroup.github.io/competitions/mrnet/ |
| MURA | primary | 40,561 radiographs, 14,863 studies, 12,173 patients; 7 upper-extremity regions | Stanford AIMI | https://stanfordmlgroup.github.io/competitions/mura/ |
| OAI | cross-domain transfer only | 4,796 participants (45-79 y), 2004-2006, 4 centers; knee MRI 3.0T + radiographs + Kellgren-Lawrence | NIH data use agreement | https://nda.nih.gov/oai |
| VQA-RAD | supplementary | 315 radiology images, 3,515 QA pairs | per source | https://osf.io/89kps/ |
| SLAKE | supplementary | 642 radiology images, 14,028 QA pairs | per source | https://www.med-vqa.com/slake/ |

Source: Datasets §; Data availability §. No new data were generated. Confidence: HIGH.

## 7. compute_target

Single NVIDIA A100 80GB. Wall-clock: ~4 h MRNet, ~12 h MURA, ~2 h per VQA dataset. Inference: ~1.2 s per
case with the 7B backbone at T=10 MC-Dropout passes (SNGP adds ~120 ms); ~480 ms per case for the
BiomedCLIP configuration. Source: Implementation details p23. Confidence: HIGH.

## 8. hparams_reference

Implementation details (p23) plus the eleven numbered equations in Methods and Tables 1-3.
AdamW, lr 2e-5, weight decay 1e-2, cosine annealing over 50 epochs with 5-epoch linear warmup;
batch size 16 (MRNet/MURA), 32 (VQA); images 224x224 with horizontal flip, +/-15 deg rotation, color
jitter; MC-Dropout T=10, p=0.1; SNGP 1,024 random Fourier features; prototype temperature init 0.07;
loss weights lambda_cal=0.1, lambda_dir=0.01; conformal lambda_unc=0.5; ECE with 15 bins; 10 seeds
(42, 123, 256, 314, 512, 628, 731, 845, 937, 1024); conformal calibration set = 20% of validation.
Confidence: HIGH.

## 9. extra_signals

- Backbone LLaVA-Med 7B (CLIP-ViT-L/14 vision encoder, d=1024; LLaMA-7B decoder, d=4096; ~7.1B total),
  frozen, adapted with LoRA r=16 on decoder attention. Cross-architecture validation with BiomedCLIP
  (ViT-B/16, d=768, PubMedBERT text encoder, ~86M). Total trainable ~3.8M (~0.05%).
- Eleven numbered equations (1-11); no algorithm boxes.
- The manuscript "Code availability" heading is present but empty, so this README carries no verbatim
  code-availability quote; checkpoints are described as available upon reasonable request.
- No fairness/ethics section imposing specific code-level demands; Acknowledgements and Author
  contributions only.
- Reported headline numbers used as evaluation targets: MRNet abnormality AUC 0.960+/-0.004, ACL
  0.977+/-0.003, meniscal 0.887+/-0.005; MURA AUROC 0.951+/-0.003; ECE 0.028+/-0.002; Brier
  0.038+/-0.003; few-shot ACL 16-shot AUC 0.941; interaction ratio UQ x Fusion = 1.31; epistemic-
  prevalence Spearman rho = -0.67; cross-modality uncertainty r = 0.72; OAI cross-domain coverage 91.3%.

## 10. Defaulted ambiguities

1. Conformal target coverage. Methods (Eq 11) and Fig. 1 mention alpha = 0.05; Results report coverage
   "above the specified 90% threshold", implying alpha = 0.10. We default to alpha = 0.10 to match the
   reported empirical coverage and record the figure discrepancy in docs/deviations.md.
2. MRI center-slice count. "the three center slices from each orientation" is read as 3 slices per
   orientation across sagittal/coronal/axial (9 total); configurable.
3. Training precision. Not reported; defaulted to bf16 on A100.
