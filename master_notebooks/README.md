# Master Notebooks

This folder is the source of truth for workshop notebooks.

Humans and AI agents edit only the files in this folder. The numbered `.py` and
`.ipynb` files at the repo root are generated student copies.

Generate all student copies:

```sh
./sync_notebooks.sh
```

To generate one notebook:

```sh
./sync_notebooks.sh 00_state_space_intuition.py
```

Run a master script directly from the repo root with `PYTHONPATH=.`:

```sh
PYTHONPATH=. uv run python master_notebooks/00_state_space_intuition.py
```

Do not hand-edit generated root notebooks unless explicitly asked.

## Exercise Markers

Use solution blocks when the master should run with answers but the student copy
should contain a stub:

```python
# BEGIN SOLUTION
b = np.log(2)
# END SOLUTION
```

The generated student copy becomes:

```python
# YOUR CODE HERE
raise NotImplementedError()
```

Use hidden-test blocks for checks that must not reach students:

```python
# BEGIN HIDDEN TESTS
assert np.isclose(np.exp(b), 2.0)
# END HIDDEN TESTS
```

Keep examples small. If an exercise needs more machinery than these markers,
move the plumbing into `code_pack`.
