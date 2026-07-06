# Notebook Reorganization Design

Date: 2026-07-06
Branch: `notebook-reorg`

## Goal

Reorganize the workshop notebooks into cognitively digestible tutorial material
for Cajal course students. Success criteria:

1. Notebooks follow the `CONTRIBUTING.md` principles: one idea per notebook,
   ~20-30 minutes each, boilerplate hidden in `code_pack`, inline
   micro-exercises with collapsed solutions, "you can now..." endings.
2. Advanced material lives in clearly-marked optional companion notebooks
   (choose-your-own-adventure branch points), with a roadmap tying the paths.
3. Notebooks run top-to-bottom in reasonable time: heavy training loads a
   pretrained checkpoint by default, with an optional "train from scratch" cell.
4. Notebook content is correct and synchronized with the lecture notes.
5. Where the lecture notes lack the underlying math, new note sections are
   written.

## Current state (from repo map)

Five paired py:percent notebooks. Two are far too long for one sitting:
`01_latent_variable_models.py` (679 lines: PCA + FA + Kalman + Ho-Kalman + EM)
and `04_XFADS_mc_maze.py` (1008 lines: fit + inference + decoding + evaluation).
Training in 01/03/04 is already gated behind `train_from_scratch = False`
(checkpoint load by default). Notebook 00 is Poisson-observation intuition;
02 is a self-contained VI demo. Lecture notes cover only linear-Gaussian models
(PPCA, FA, rotation, scale, Kalman filter, Ho-Kalman).

## New notebook sequence (renumbered, flat)

Sequential numbering. Each notebook's role (core vs optional deep dive) is marked
in its title and in the roadmap. The fast track is the core set; optional
notebooks sit right after the core topic they extend, as branch points.

| # | Notebook | Role | Source |
|---|----------|------|--------|
| 00 | `00_state_space_intuition` | core | 00 (tightened) |
| 01 | `01_snr_and_readout_geometry` | optional (extends 00) | 00 (SNR + geometry cells) |
| 02 | `02_linear_lvms_pca_fa_kalman` | core | 01 (PCA/FA/Kalman/RTS) |
| 03 | `03_system_id_and_em` | optional (extends 02) | 01 (Ho-Kalman + EM) |
| 04 | `04_variational_inference` | core | 02 |
| 05 | `05_xfads_ring_attractor` | core | 03 |
| 06 | `06_xfads_mc_maze` | core | 04 (fit + inference) |
| 07 | `07_decoding_and_evaluation` | optional (extends 06) | 04 (decoding/eval) |

Fast track (core): 00 -> 02 -> 04 -> 05 -> 06. Optional branches: 01, 03, 07.

Each core notebook ends with an explicit branch point: a pointer to its optional
deep dive and to the next core notebook. A roadmap in `README.md` and a header
cell in `00` draw both paths. Colab badge URLs are updated to the new filenames
(the renumbering breaks old links - accepted tradeoff).

## Per-notebook plan

- **00 state space intuition (core):** keep the generative Poisson-spike story;
  move SNR and random-vs-axis-aligned readout to 01. Add a short Poisson
  observation reminder linking the new lecture-note section. Add 1-2 inline
  micro-exercises and a "you can now..." ending.
- **01 SNR and readout geometry (optional):** SNR/Fisher-information and readout
  geometry pulled from 00. Links to the new Poisson observation note section.
- **02 linear LVMs (core):** simulate the spiral LDS, then PCA -> FA ->
  Kalman/RTS smoothing -> comparison. Maps to notes sec:ppca, sec:fa,
  sec:rotation, sec:scale, sec:kalman. Trim the long inline math (it now lives in
  the notes; link instead). Add exercises.
- **03 system ID and EM (optional):** Ho-Kalman subspace ID + one EM step, the
  heavy math cell. Maps to notes sec:hokalman. Links to notes for derivations.
- **04 variational inference (core):** keep the from-scratch ELBO demo; add
  exercises. Links to the new VI note section.
- **05 XFADS ring attractor (core):** end-to-end XFADS on synthetic data,
  checkpoint by default. Clean the stray "python / Copy / Edit" paste artifacts
  (03 lines ~338-343). Links to the new XFADS conceptual note section.
- **06 XFADS mc_maze (core):** load data, build model, checkpoint, then
  smoothing / filtering / forecasting and reconstructed firing rates. The core
  inference story on real data.
- **07 decoding and evaluation (optional):** ridge decoding R^2, k-step-ahead
  prediction, PCA-vs-R^2 sweep, predictive log-likelihood. The heavier
  evaluation compute.

## New lecture-note sections

Depth: full derivations where the math is self-contained; conceptual (cited) for
XFADS/VAE. New sections, in the notes' existing style (`\ifsolutions` exercises
where natural):

1. **Poisson / exponential-family observations** (full) - used by 00 and 06.
   The `lambda = exp(Cx + b)` model, log-likelihood, why the Gaussian machinery
   generalizes.
2. **RTS smoothing and forecasting** (full) - extend the Kalman section; the
   notes currently stop at the forward filter.
3. **Variational inference: ELBO, KL, reparameterization** (full) - the major
   gap; the conceptual bridge to XFADS.
4. **Amortized inference / VAE / recognition networks** (conceptual, cited).
5. **Nonlinear structured state-space models / XFADS** (conceptual, cited) -
   architecture, structured Gaussian posterior, natural-parameter additive
   decomposition, with references.

Each new notebook section links to its note section and vice versa (one
canonical home for each derivation; no copy-paste, per CONTRIBUTING).

## Correctness and runtime verification

- Each core notebook must run top-to-bottom against checkpoints in reasonable
  time; verify by executing the paired script.
- Cross-check every equation/claim in the notebooks against the corresponding
  lecture-note section; flag and fix discrepancies (notation, signs, model
  definitions). Notation must match across notes and notebooks (`x`, `y`, `C`,
  `lam`, etc.).
- Keep `.py` and `.ipynb` in sync via jupytext for every changed notebook.

## Out of scope

- Retraining models / producing new checkpoints (use existing `ckpts/`).
- Changing the XFADS library itself.
- New datasets.

## Execution note (ultracode)

Implementation will be orchestrated with workflows: a verify/analyze phase
(per-notebook correctness + note-sync audit), a content phase (splits, exercises,
new note sections), and a final run-through phase (execute each core notebook,
confirm runtime). Each phase's results are reviewed before the next.
