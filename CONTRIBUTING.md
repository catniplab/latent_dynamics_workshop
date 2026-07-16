# Contributing to the Latent Dynamics Workshop

This is a teaching repository for motivated masters-level students who know some
math and want to apply these methods to their own data.

Use the working-memory budget for the concept, not scaffolding. Hide boilerplate,
plotting, data wrangling, and setup unless they are the point of the lesson.
Reduce learners' cognitive load with a consistent experience: notebooks should
feel like parts of one course, not separate demos with different conventions.

---

## Authoring model

**The only editable notebook sources are `master_notebooks/*.py`. Never edit the
root-level `NN_*.py` or `NN_*.ipynb` by hand - they are generated and your edits
will be overwritten on the next sync.**

Workflow:

1. Edit `master_notebooks/NN_*.py`.
2. Run `./sync_notebooks.sh [name.py]` to regenerate the root-level student `.py`
   and `.ipynb`. The sync strips `# BEGIN/END SOLUTION` and `# BEGIN/END HIDDEN
   TESTS` blocks, replacing solutions with `# YOUR CODE HERE` and
   `raise NotImplementedError()`.
3. Commit the master source and the generated student copies together.

To execute a master `.py` directly, run it from the repo root with `PYTHONPATH=.`.

---

## Launch reliability

The first requirement is simple: every notebook must launch for someone who is
not you, on a machine that is not yours.

Colab is the strictest target. Opening a badge loads only that `.ipynb`; the repo
is not cloned, `code_pack` and `external/` are absent, and the working directory
is not the repo. Any notebook that imports local code or reads repo files must
bootstrap itself before the first real import or file read.

### Startup checklist

Run this for every notebook that imports beyond the standard scientific stack or
reads a file:

- [ ] Opens from the Colab badge and runs top-to-bottom on a fresh runtime.
- [ ] Has a Setup cell that clones the repo when `_in_colab`.
- [ ] Uses `--recurse-submodules` only when it needs `external/`.
- [ ] Makes imports resolvable: `chdir` into the clone for repo-relative files,
      or append the repo root/submodule dirs to `sys.path`.
- [ ] Depends only on public submodules; unauthenticated recursive clone must
      work.
- [ ] Uses file paths that resolve from the runtime working directory.
- [ ] Verifies live `!` shell lines in the `.ipynb`, not only the `.py`.
- [ ] Badge and Setup clone both target the branch students use, usually `main`.
- [ ] Uses environment-agnostic rendering, e.g. `tqdm.auto` instead of
      notebook-only widgets.

### Known launch failures

- **Missing Setup cell.** `00` / `01` imported `code_pack` locally and failed in
  Colab. Any `code_pack` or `external/` import needs bootstrap.
- **Wrong cwd.** Repo-relative paths such as `vanderpol/data/...` and
  `!python code_pack/...` fail under the badge unless the notebook enters the
  clone or resolves paths explicitly.
- **Script magics are misleading.** `# !git clone ...` in a jupytext `.py` is a
  live shell command in the `.ipynb`; check the actual notebook.
- **Badges track `main`.** Feature-branch notebook fixes are invisible to
  students until merged, and the bootstrap clone usually pulls `main`.

### Changing startup code

Treat a working Setup cell, import path, and file path as a contract:

- Change bootstrap only with a reason and a fresh Colab re-test.
- Prefer the smallest edit that fixes the reported problem.
- For new notebooks, copy a proven pattern: `02` for submodule + `sys.path`,
  `00` for `chdir` + repo-relative data.

---

## Notebook principles

1. **One notebook = one idea.** If the takeaway is not one sentence, split it.
2. **Text serves the next cell.** Markdown should fit on screen and point to what
   the student is about to run or see. Put deep derivations in notes or appendices.
3. **Equation next to code.** Put each key equation immediately above its
   implementation, using matching names (`lam`, `C`, etc.).
4. **Hide boilerplate.** Plotting, loading, repeated setup, and long non-concept
   cells belong in `code_pack`.
5. **Runnable on arrival.** The authored notebook runs top-to-bottom with no
   manual edits, and the Colab badge works.
6. **Object before theory.** Show the spike train, trajectory, or phenomenon
   before formalizing the model.
7. **One active step per concept.** Each concept needs at least one prediction,
   implementation, or written claim. Looking at output does not count.
8. **Keep the learner experience consistent.** Reduce cognitive load by reusing
   notation, section order, exercise style, visual conventions, and helper APIs.
   Keep `z` as latent, `y` as observations, `C` as loading, and recurring helper
   signatures stable.
9. **Progressive disclosure.** Do not introduce a new concept, tool, and dataset
   in the same cell.
10. **End with "you can now...".** Close with the gained capability and one
    transfer prompt.

---

## Notebook template

Use this section order:

1. Title + Colab badge + one-sentence takeaway.
2. Hook / visual: generate and show the object of interest.
3. Reminder: minimal math recap, not a full derivation.
4. Equation.
5. Implementation directly below the equation, with matching names.
6. Inline micro-exercises.
7. "You can now..." capability statement + transfer prompt.

A skeleton `.py` belongs in `master_notebooks/_template.py`; create it from an
existing notebook if missing.

---

## Exercises

Use 1-2 inline micro-exercises per notebook, placed where the concept lives.
Avoid the fluency illusion: scrolling a finished notebook with pre-rendered
outputs feels like learning, but is not. Ship notebooks with outputs cleared,
and require a written prediction or claim before the confirming run.

Exercise forms:

- **Predict, then run.** Put a Markdown prompt immediately before the revealing
  cell, with a visible answer slot:

  ```markdown
  Before running: as `noise_std` goes `0.1 -> 5.0`, what happens to the width of
  the confidence interval? Does coverage change? Write one sentence.

  Your prediction:
  ```

- **Fill a small stub.** The student writes the concept, not the scaffolding:

  ```python
  def log_likelihood(x, mu, sigma):
      """
      x : (N,) array of observations
      returns: scalar log p(x | mu, sigma)
      """
      # ============ YOUR CODE HERE (about 3 lines) ============
      raise NotImplementedError
      # ========================================================
  ```

  Mark scaffold clearly:

  ```python
  # ---- PROVIDED: do not edit ----
  def plot_posterior(samples, ax=None):
      ...
  ```

  Then put a self-check cell immediately below:

  ```python
  assert np.isclose(log_likelihood(x, 2.0, 1.0), -3.7568, atol=1e-3)
  print("passed!")
  ```

- **Tweak with a target.** Do not ask students to "explore" a knob. Give a
  falsifiable endpoint. Breaking things is often the best target: the failure
  mode is the concept.

  ```text
  Find the smallest n where the CLT approximation is visually acceptable; then
  find a distribution where that answer is wrong by 10x.

  Break it: find parameters where the estimator is confidently wrong, and say why.

  Two of these three settings give the same MSE for different reasons. Which two,
  and what are the reasons?
  ```

Rules:

- One idea per exercise.
- No unrelated boilerplate or scavenger hunts.
- The cell must run once the missing line is filled.
- Non-stretch exercises alone must cover every concept's active step.
- Mark harder prompts with `> **Stretch (optional):**`.
- Give 3-4 specific values instead of free rein, chosen to straddle a qualitative
  transition, e.g. `10, 50, 500`.
- Use sliders only for fast intuition about a smooth relationship. If the goal
  is reasoning about mechanism, make students retype discrete values.
- Require an artifact: a filled Markdown sentence, claim, or explanation. A plot
  alone is not evidence of understanding.

Generated student fill-in exercises may break top-to-bottom execution until
filled. The master solution state must run clean. Keep answers in
`# BEGIN SOLUTION` / `# END SOLUTION` blocks so `./sync_notebooks.sh` replaces
them with `# YOUR CODE HERE` and `raise NotImplementedError()` in student
copies. Put tests that should not reach students in `# BEGIN HIDDEN TESTS` /
`# END HIDDEN TESTS` blocks.

Do not ship separate solution notebooks or a solutions branch.

---

## Code visibility and helpers

Remove redundant code without hiding the concept.

1. **Split concept from plumbing.** Concept code stays visible. Plumbing
   (plotting, raster assembly, data loading, environment setup) goes to
   `code_pack`.
2. **Teach first, hide later.** Show a pattern in full the first time it is the
   concept; use helpers for later reuse.
3. **Name helpers by intent.** `plot_raster(y)` and `simulate_poisson(lam, dt)`
   should be understandable without opening the helper.
4. **Keep helper APIs small and stable.** A shared helper must recur enough to
   earn its place.
5. **Do not abstract one-offs.** Leave single-use plumbing inline with a short
   comment.

Rule of thumb: if hiding code forces a student to open `code_pack`, rename the
helper or leave the code visible.

---

## Length

Target one idea and roughly 20-30 minutes. There is no strict cell count. If the
single takeaway is hard to state, split the notebook.

---

## Lecture notes

Lecture notes are the compact companion to the notebooks, not a textbook.

1. **One idea per section.** Section headings should map to notebook takeaways.
2. **Share notation with code.** Use the same symbols and variable names:
   `z`, `y`, `C`, `lam`.
3. **Notes derive; notebooks do.** Put derivations in notes once, with only the
   shortest honest path to the result. Put tangents in `Optional` sections.
4. **Object before theory.** Start from the plot, phenomenon, or question.
5. **Point to the active step.** End each section with an explicit
   `-> see notebook NN` hook.
6. **Avoid duplication.** Equations should have one canonical home. Update notes
   before implementation when equations change.
7. **Keep disclosure progressive.** Reuse vocabulary and introduce one new idea
   at a time.

Keep notes beside notebooks so notation drift is caught in review.

### Exercises in notes

Exercises live inline in the `.tex`; each solution sits immediately after its
exercise and is hidden by default via `\ifsolutions` in `math_preamble.tex`.

```latex
\begin{exercise}
Show that as $\sigma^2 \to 0$ the posterior mean approaches the least-squares
projection.
\end{exercise}
\begin{solution}
As $\sigma^2 \to 0$, $M \to C\trp C$, so $M^{-1}C\trp\vy \to (C\trp C)^{-1}C\trp\vy$.
\end{solution}

\begin{exercise}[optional stretch]
...
\end{exercise}
```

Rules:

- Inline, one idea each; the standalone `kalman_filter_exercise.tex` worksheet
  is the exception.
- Every `exercise` gets a following `solution`.
- Harder exercises use `[optional stretch]`.
- Non-stretch exercises alone must cover the section's active step.

Build from `cd lectures`:

```sh
make student     # latent_dynamics_notes.pdf, solutions hidden
make solutions   # latent_dynamics_notes_solutions.pdf, solutions shown
make exercise    # kalman_filter_exercise.pdf
make             # all of the above
```

PDFs are gitignored. Compile and commit PDFs only for a version release.

---

## Review checklist

Before merging a notebook, confirm:

- [ ] Takeaway statable in one sentence.
- [ ] Runs top-to-bottom clean, no manual edits, Colab badge works.
- [ ] Notebook outputs are cleared before shipping.
- [ ] Every key equation has adjacent, name-matched implementation.
- [ ] No visible non-concept cell over ~15 lines.
- [ ] Every concept has at least one non-stretch micro-exercise.
- [ ] Prediction/tweak exercises require a written answer before the reveal.
- [ ] Stub exercises mark provided code and include an immediate assert check.
- [ ] Tweak exercises have bounded values and a falsifiable target.
- [ ] Harder exercises are marked optional stretch.
- [ ] Fill-in answers live in master-only solution blocks.
- [ ] Only unsolved fill-in cells error; predict/tweak prompts stay runnable.
- [ ] Plumbing is not duplicated where a named helper should hold it.
- [ ] Ends with "you can now..." plus a transfer prompt.
- [ ] Learner experience matches the rest of the sequence: notation, section
      order, exercise style, visuals, and helper APIs.
- [ ] Master source and generated student `.py`/`.ipynb` copies are committed
      and in sync.
