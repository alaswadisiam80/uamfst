# PATENT APPLICATION

**Title:** Uncertainty-Aware Multimodal Few-Shot Transfer for Musculoskeletal Injury Diagnosis and
Clinical Decision Support (UA-MFST)

| Field | Entry |
|---|---|
| Application No. | UAMFST/0.1.0 |
| Filed | 2026-05-29 |
| Int. Cl. | G06N 3/02 (neural networks); G16H 50/20 (medical diagnosis computing) |
| Applicant | Authors of the npj Digital Medicine manuscript of the same title |
| Status | Working embodiment; analytic substrate runs on CPU, reported figures need the licensed backbone and data |

---

## ABSTRACT

A system and method are disclosed for trustworthy multimodal injury diagnosis. A frozen vision-language
backbone is wrapped with five cooperating parts: (a) a spectral-normalized neural Gaussian process
together with a Monte-Carlo dropout estimator yielding an epistemic uncertainty; (b) a Dirichlet output
head yielding an aleatoric uncertainty; (c) a fusion gate that reweights the image and text channels by
per-sample epistemic precision; (d) an uncertainty-weighted few-shot prototype adapter; and (e) a
split-conformal selective-deferral rule giving finite-sample coverage. The embodiment attains the
manuscript figures on MRNet and MURA while exposing calibrated confidence and a principled abstention
option for clinical use.

---

## FIELD OF THE INVENTION

The invention relates to machine learning for medical imaging, and more particularly to multimodal large
language models that quantify and act upon predictive uncertainty when diagnosing musculoskeletal and
sports injuries from radiographs and magnetic-resonance studies paired with structured clinical text.

## BACKGROUND OF THE INVENTION

Prior multimodal classifiers report a single point prediction without a faithful account of epistemic
(model) and aleatoric (data) uncertainty, and without a guarantee on the error rate of the cases they
choose to answer. In a clinical workflow this makes silent over-confidence hard to detect and gives no
defensible rule for referring hard cases to a human reader. A diagnosis system therefore benefits from
calibrated uncertainty, a means to learn new tasks from few labelled examples, and a coverage guarantee
on retained predictions.

## SUMMARY OF THE INVENTION

The disclosed embodiment composes a frozen backbone with the five parts named in the abstract. Epistemic
uncertainty is obtained in closed form from a random-feature Gaussian process and refined by Monte-Carlo
dropout; aleatoric uncertainty is read from a Dirichlet head. A fusion gate forms a precision-weighted
combination of the image and text channels. A few-shot adapter builds class prototypes weighted by the
inverse epistemic variance of each support example. A split-conformal procedure converts the calibrated
scores into prediction sets and defers a case when the set is too large or its epistemic variance exceeds
a calibrated threshold. Training minimizes a composite objective combining cross-entropy, a differentiable
calibration term, and a Dirichlet-to-uniform regularizer.

---

## BRIEF DESCRIPTION OF THE DRAWINGS

FIG. 1 depicts the estate pipeline of the preferred embodiment, organized as a working winery.

```
   vineyard ---> pressing ---> fermentation ---> maceration ---> fining ---> appellation
   (datasets)   (backbone,     (SNGP yeast +     (mc-dropout,    (ECE,        (split-conformal
                 DCT must,       Dirichlet         epistemic       composite     prediction set +
                 projection)     malolactic)       tannins,        objective,    declassification /
                                                   fusion          temperature)  deferral)
                                                   marrying)
                          \                                   /
                           grafting (few-shot scions)        /
                                       |                     |
                  cuverie (manifold gin shim + cuvee = full UAMFST + assemblage)
                                       |
        maturation (aging loop) --- tasting (palate / panel / judging) --- cooperage (logbook / casks)
                                       |
                            cellarmaster (vintage / scoring / pour / label)
```

FIG. 2 (conceptual) shows uncertainty heterogeneity across cases and the reliability of the calibrated
confidence; its data is produced by `tasting/flight.py` and `tasting/panel.py`.

---

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

The package keeps the import name `uamfst` and a `src/` layout. Its subpackages are named after the
stages of a winery; the equation-to-code map is in `docs/implementation-map.md`.

| Stage (subpackage) | Role in the embodiment |
|---|---|
| `charter` | typed contracts (`covenants`) and frozen dataclass records (`vessels`) |
| `vineyard` | dataset adapters (`mrnet`, `mura`, `oai`, `vqa`), `harvest`, `parcel`, `consignment`, `sorting`, `nursery` |
| `pressing` | frozen backbones (`varietals`), analytic substrate (`must`), projection (`extraction`), tri-slice (`crushing`), text (`bouquet`), low-rank adapter (`fortification`) |
| `fermentation` | random-feature Gaussian process (`yeast`) and Dirichlet head (`malolactic`) |
| `maceration` | Monte-Carlo dropout (`punchdown`), epistemic combiner (`tannins`), fusion gate (`marrying`) |
| `grafting` | uncertainty-weighted few-shot prototypes (`scions`) |
| `fining` | calibration error (`clarity`), composite objective (`adjustments`), temperature scaling (`cold_stabilization`), reliability (`stability`) |
| `appellation` | split-conformal sets (`controlled_origin`) and dual-criterion deferral (`declassification`) |
| `cuverie` | typed gin shim (`manifold`), full estimator (`cuvee` = UAMFST), registration (`assemblage`) |
| `maturation` | training loop (`aging`), optimizer/schedule (`racking`), EMA (`lees`), checkpoint (`bottling`), seeding (`inoculation`) |
| `tasting` | metrics (`palate`), statistics (`panel`), interaction (`balance`), selective (`flight`), evaluator (`judging`) |
| `cooperage` | logging (`logbook`), distributed helpers (`crew`), I/O (`casks`) |
| `cellarmaster` | absl entry points: `vintage` (train), `scoring` (evaluate), `pour` (infer), `label` (export), `station` (runtime) |

**Reduction to practice (installation).** The default analytic backbone keeps everything CPU-runnable;
the real backbone and licensed datasets are only needed for the reported numbers.

```text
python -m pip install -r requirements.txt
python -m pip install -e .
```

conda and Docker are equivalent entry points:

```text
conda env create -f environment.yml && conda activate uamfst && python -m pip install -e .
docker build -t uamfst:latest . && docker run --gpus all uamfst:latest \
  uamfst.cellarmaster.vintage --gin_file configs/experiment/main.gin
```

Extras: `.[backbones]` adds the real LLaVA-Med / BiomedCLIP weights stack, `.[export]` adds the ONNX
exporter, `.[dev]` adds lint, type, and test tooling. A self-check of the embodiment:

```text
python -m pytest -q        ->   51 passed
```

**Materials (datasets).** `bash scripts/prepare_data.sh data` prints the expected on-disk layout.
Preprocessing is fitted on training data and frozen before test evaluation; each adapter records the
SHA-256 of its frozen manifest via `uamfst.vineyard.consignment.manifest_sha256`.

| dataset | role | size | license | source |
|---|---|---|---|---|
| MRNet | primary | 1,370 knee MRI exams (1,250 patients), split 1,130 / 120 / 120 | Stanford AIMI | https://stanfordmlgroup.github.io/competitions/mrnet/ |
| MURA | primary | 40,561 radiographs, 7 upper-extremity regions | Stanford AIMI | https://stanfordmlgroup.github.io/competitions/mura/ |
| OAI | cross-domain transfer | 4,796 participants, knee MRI + radiographs + KL grade | NIH data use agreement | https://nda.nih.gov/oai |
| VQA-RAD | supplementary | 315 images, 3,515 QA pairs | per source | https://osf.io/89kps/ |
| SLAKE | supplementary | 642 images, 14,028 QA pairs | per source | https://www.med-vqa.com/slake/ |

**Training the preferred embodiment.** This reads the manuscript hyperparameters from
`configs/train/schedule.gin` (AdamW, lr 2e-5, 50 epochs, batch 16, bf16).

```text
python -m uamfst.cellarmaster.vintage --gin_file configs/experiment/main.gin
```

**Limitations of the embodiment.**

- The default backbone is a deterministic analytic substrate for CPU development and testing; the
  reported numbers need the real LLaVA-Med weights and the licensed datasets.
- MRNet and MURA are single-centre (Stanford); cross-centre robustness within the same task is
  unverified and would require per-site calibration.
- Conformal target coverage defaults to alpha = 0.10 (90%), matching the reported empirical coverage;
  `docs/deviations.md` notes the figure discrepancy.

---

## WORKING EXAMPLES

Each example is a command and the manuscript value with its tolerance (mean +/- std over 10 seeds; AUC
differences via DeLong with Holm-Bonferroni over 12 comparisons). Append `--checkpoint <run>/model.ckpt`
to evaluate a trained run. The evaluator is invoked as
`python -m uamfst.cellarmaster.scoring` (or the installed `uamfst-evaluate` command, shown as `evaluate`).

| # | endpoint | command | expected |
|---|---|---|---|
| 1 | MRNet abnormality AUC | `evaluate --gin_file configs/experiment/main.gin` | 0.960 +/- 0.004 |
| 2 | MRNet ACL AUC | same, `--gin_binding 'MRNetDataset.task="acl"'` | 0.977 +/- 0.003 |
| 3 | MRNet meniscal AUC | same, `--gin_binding 'MRNetDataset.task="meniscus"'` | 0.887 +/- 0.005 |
| 4 | MURA AUROC | `evaluate --gin_file configs/experiment/main_mura.gin` | 0.951 +/- 0.003 |
| 5 | Brier score | any imaging endpoint above | 0.038 +/- 0.003 |
| 6 | Few-shot ACL, 16-shot AUC | train then evaluate with a 16-shot support set | 0.941 |
| 7 | BiomedCLIP MRNet abnormality AUC | `evaluate --gin_file configs/experiment/supplementary_biomedclip.gin` | 0.956 +/- 0.004 |
| 8 | BiomedCLIP ECE | same | 0.032 +/- 0.002 |
| 9 | Deferral error reduction at 15% | reported by `evaluate` | 44% (7.7% -> 4.3%) |
| 10 | OAI cross-domain coverage | `evaluate --gin_file configs/experiment/supplementary_oai_transfer.gin` | 91.3% |

The fusion interaction ratio (1.31) is recomputed from the ablation AUCs by
`uamfst.tasting.balance.interaction_ratio`; the Table 2 ablations live in
`configs/experiment/ablation_*.gin`.

---

## INDUSTRIAL APPLICABILITY

The embodiment runs on a single accelerator and adds a small trainable footprint over a frozen backbone.

| item | value |
|---|---|
| accelerator | 1x NVIDIA A100 80GB |
| wall-clock | ~4 h MRNet, ~12 h MURA, ~2 h per VQA dataset |
| inference | ~1.2 s per case (LLaVA-Med 7B, T=10 MC-Dropout); ~480 ms (BiomedCLIP) |
| trainable parameters | ~3.8M (~0.05% of the 7.1B backbone) |
| precision | bf16 |

---

## CLAIMS

What is claimed is:

1. A system for uncertainty-aware multimodal diagnosis of musculoskeletal injury, comprising: a frozen
   vision-language backbone producing image and text representations; an epistemic estimator comprising a
   spectral-normalized random-feature Gaussian process and a Monte-Carlo dropout sampler; a Dirichlet
   output head producing an aleatoric uncertainty; a fusion gate forming a precision-weighted combination
   of the image and text representations; and a selective-deferral module producing a calibrated decision
   or an abstention.

2. The system of claim 1, wherein the epistemic estimator combines a closed-form Gaussian-process variance
   with a Monte-Carlo dropout sample variance over T stochastic passes.

3. The system of claim 1, further comprising a few-shot adapter that constructs class prototypes weighted
   by the inverse epistemic variance of each support example, and classifies a query by a temperature-scaled
   nearest-prototype softmax.

4. The system of claim 1, wherein the selective-deferral module forms a split-conformal prediction set with
   finite-sample coverage at level 1 - alpha and defers a case when the set size exceeds a limit or the
   epistemic variance exceeds a calibrated percentile threshold.

5. The system of claim 1, wherein the frozen backbone comprises one of a LLaVA-Med model and a BiomedCLIP
   model, and wherein a deterministic discrete-cosine-transform substrate stands in for the backbone for
   computation without trained weights.

6. A method of training the system of claim 1, comprising minimizing a composite objective equal to a
   cross-entropy term plus a weighted differentiable expected-calibration-error term plus a weighted
   Kullback-Leibler divergence from the Dirichlet output to a uniform Dirichlet.

7. The method of claim 6, wherein an ablation selectively disables the uncertainty estimator, the fusion
   gate, or the few-shot adapter to attribute the contribution of each.

---

## STATEMENT REGARDING CODE AND DATA AVAILABILITY

Trained weights are available upon reasonable request; a released checkpoint will carry a SHA-256
manifest matching the preprocessing hash recorded at evaluation time. All datasets are publicly
available at the sources above and no new data were generated; the manuscript carries no separate
code-availability statement.

---

## CITATION

```bibtex
@article{uamfst2025,
  title   = {Trustworthy Multimodal Large Language Models with Uncertainty-Aware Few-Shot Transfer
             Learning for Sports Injury Diagnosis and Clinical Decision Support},
  journal = {npj Digital Medicine},
  year    = {2025}
}
```
