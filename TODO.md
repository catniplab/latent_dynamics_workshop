# TODO

## 01 - SNR and readout geometry

- Move the axis-aligned vs random-projection readout demo here (removed from
  `00` on 2026-07-12, where it was mislabeled as an inverse-link effect). `01` is
  the notebook that owns readout geometry. It currently rebuilds its generative
  model in memory, so add the axis-aligned loading (block-diagonal `C_tilde`) and
  its raster there rather than reading from the h5. The old generator code for
  this - `generate_poisson_observations_axis_aligned` and the `Y_axis`/`C_tilde`
  datasets - was dropped from `code_pack/generate_vdp_data.py`; recover it from
  git history if useful.
