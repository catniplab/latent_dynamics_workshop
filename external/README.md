# External dependencies (git submodules)

Third-party packages used by the workshop notebooks. Initialize with:

```bash
git submodule update --init --recursive
```

| Submodule | Package | Used in |
|-----------|---------|---------|
| `xfads/` | XFADS state-space modeling | notebooks 01, 03, 04 |
| `neurofisherSNR/` | Fisher-information SNR bounds | notebook 00 |
| `nlb_tools/` | Neural Latents Benchmark data tools | `mc_maze/data_preprocessing.py` |

`xfads` and `neurofisherSNR` are installed editable by the environment: `uv sync`
reads `pyproject.toml` (see `[tool.uv.sources]`), and the conda fallback installs
them via `env.yml`. `nlb_tools` is added to the Python path where needed.
