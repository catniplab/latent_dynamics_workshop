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

Installed editable via `env.yml` (`xfads`, `neurofisherSNR`). `nlb_tools` is on the Python path where needed.
