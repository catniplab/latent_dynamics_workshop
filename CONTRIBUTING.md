# Contributing to the Latent Dynamics Workshop

This is a teaching repository. The audience is highly motivated masters-level
students with limited cognitive bandwidth for long reading and long code, who
know some math but benefit from reminders, and whose end goal is to **apply**
these methods to their own data.

Every notebook competes for a fixed working-memory budget. The single rule
behind everything below: **spend that budget on the concept, never on the
scaffolding.** Boilerplate, plotting code, data wrangling, and unexplained
magic are load leaks. Push them out of sight so the idea is what remains.

---

## Authoring model

Notebooks are paired to `py:percent` scripts via jupytext (see `jupytext.toml`).
**Edit the `.py` file**, then let jupytext sync the `.ipynb`. Do not hand-edit
both. Commit both the `.py` and the `.ipynb`.

---

## The ten principles

1. **One notebook = one idea.** If you cannot state the takeaway in one
   sentence, split the notebook. Each file introduces exactly one genuinely new
   element and reuses everything prior.

2. **Text serves the next cell.** Markdown cells are short (fits on screen
   without scrolling, roughly 5-8 lines) and always point forward to something
   the student is about to run or see. Deep derivations go to a linked appendix
   or reference, not into the flow.

3. **Equation next to code, with matching names.** Every key equation sits
   immediately above the cell that implements it, and variable names mirror the
   math (`lam` for the rate, `C` for the loading matrix). Never let math and its
   implementation drift more than one cell apart.

4. **Hide boilerplate, expose the concept.** Plotting, data loading, and
   repeated setup live in `code_pack` and are imported. A visible cell longer
   than ~15 lines that is not itself the point belongs in a helper.

5. **Runnable on arrival.** Every notebook runs top-to-bottom with no manual
   edits. Keep the Colab badge working. The first executable cell just works.

6. **Show the object before the theory.** Generate and plot the thing (spike
   train, trajectory) first, then explain the model that produced it.

7. **One active step per concept.** Reading code is not learning to apply it.
   Each concept has at least one place where the student does something. See
   Exercises below.

8. **Consistent vocabulary and API across notebooks.** `x` is the latent, `y`
   the observations, `C` the loading, and recurring helpers keep the same
   signatures. Novelty is load; reuse is a bandwidth saver.

9. **Progressive disclosure.** Never introduce a new concept, a new tool, and a
   new dataset in the same cell.

10. **End with "you can now..."** Close each notebook with a one-line statement
    of the capability gained plus a concrete transfer prompt ("try this on your
    own data / this variation").

---

## Notebook template

Section order for every notebook:

1. **Title + Colab badge + one-sentence takeaway.**
2. **Hook / visual** - generate and show the object of interest.
3. **Reminder** - the minimal math recap needed to proceed (not a full
   derivation).
4. **Equation.**
5. **Implementation** - the equation in code, immediately below it, matching
   names.
6. **Inline micro-exercise(s)** - see below.
7. **"You can now..."** - capability statement + transfer prompt.

A skeleton `.py` is in `notebooks/_template.py` (create it from an existing
notebook if it does not exist yet).

---

## Exercises

**Format: inline micro-exercises.** Weave 1-2 small prompts into each notebook,
right where the concept lives. Keep them to one of:

- **Predict** - "before running the next cell, what happens to the raster if
  `a` doubles?"
- **Fill one line** - leave a single `# YOUR CODE HERE` inside an otherwise
  complete cell, so the student writes the line that *is* the concept and
  nothing else.
- **Tweak and observe** - "change `frq` and rerun; explain the plot."

Rules:

- One idea per exercise. Never require unrelated boilerplate to complete it.
- The surrounding cell must run once the one missing line is filled - no
  scavenger hunts across cells.
- Exercises are the mechanism for principle 7, not an afterthought. Every
  concept gets at least one.

**Fill-in exercises are allowed to break top-to-bottom execution.** A blanked
`# YOUR CODE HERE` cell will error until filled - that is intended and does not
violate principle 5. Principle 5 governs the *authored* state (all solutions
filled runs clean). The collapsed solution is what restores runnability for a
stuck student. Predict and tweak exercises, by contrast, keep the notebook
runnable throughout.

**Mark harder exercises as optional stretch.** Keep the core path short. Any
exercise beyond the minimum needed to grasp the concept gets a
`> **Stretch (optional):**` prefix so students can skip it without falling
behind. The non-stretch exercises alone must cover principle 7.

**Solutions: collapsed and in-notebook.** Put the solution directly below the
exercise, hidden by default so students self-check without leaving the notebook.
Use a Markdown `<details>` block (renders collapsed on GitHub and Colab):

```markdown
<details>
<summary>Solution</summary>

​```python
lam = np.exp(a * x + b)
​```

</details>
```

Do not ship separate solution notebooks or a solutions branch.

---

## Minimizing redundant code without hiding the concept

Redundant, repeated code across notebooks is a distraction, but the naive fix -
move everything into helpers - can *increase* load if students must open
`code_pack` to know what a helper does. The goal is to remove the redundancy
while keeping every hidden thing so obvious it never needs to be opened. Rules:

1. **Split code into "concept" and "plumbing."** Concept code is what the
   student is meant to learn here; it stays visible in the notebook. Plumbing is
   everything else - plotting, raster assembly, data loading, environment setup.
   Plumbing goes to `code_pack`. Test: *would a student learn anything by
   re-reading this code?* If no, it is plumbing.

2. **First occurrence teaches, later occurrences hide.** Show a pattern in full,
   inline, the one time it is the concept (e.g. the Poisson spike generation in
   `00`). Every later reuse calls a helper. Nothing is hidden before it has been
   taught once in the open.

3. **Name helpers so the body never needs reading.** `plot_raster(y)`,
   `simulate_poisson(lam, dt)` read as English at the call site. If a student
   would have to open the helper to understand the line, the name has failed -
   you moved load, you did not remove it. Helper = verb of intent + obvious
   signature.

4. **Keep the helper surface small and stable.** A handful of well-named
   functions reused across all notebooks beats many one-off helpers (principle
   8). A new helper must earn its place by recurring.

5. **Do not abstract single-use code.** If an operation appears once and is not
   the concept, leave it inline and add a one-line comment. A helper used in one
   place is indirection without payoff.

The line to walk: redundancy across notebooks -> promote to a well-named helper.
Redundancy within one notebook that is the concept -> keep it, it is the lesson.
One-off plumbing -> inline with a comment. When unsure, ask "does hiding this
force anyone to open `code_pack`?" If yes, either rename it or leave it visible.

## Length

Target **one idea, roughly 20-30 minutes** to work through. There is no strict
cell count. The test at review time is: can you state the single takeaway in one
sentence? If not, split. If a notebook feels long, that is a signal to break it,
not to keep scrolling.

---

## Lecture notes

The same load budget applies to prose. Lecture notes are not a textbook; they
are the minimal companion that lets a student reconstruct the idea and feed the
notebooks. Translate the principles:

1. **One idea per section.** Section headings map one-to-one onto notebook
   takeaways. A reader should be able to pair a note section with its notebook.

2. **Shared notation with the code.** The symbol in the notes is the variable in
   the notebook: `x` latent, `y` observations, `C` loading, `lam` the rate. Fix
   the notation once, up front, and never let notes and code diverge (this is
   principle 3 across media, and it is the highest-leverage rule for notes).

3. **Notes are the canonical home for the math; notebooks do, notes derive.**
   This is the division of labor: derivations the notebooks omit live here, once,
   in full (PPCA/FA/Kalman inference and learning, etc.). But "self-contained"
   is not "exhaustive" - each derivation is the shortest honest path to the
   result, motivated before it is carried out, with truly tangential algebra
   pushed to an `Optional` section (as the Ho-Kalman section already does).
   Inside a notebook, by contrast, keep to a reminder and point to the note.

4. **Object before theory.** Introduce each concept with the thing it explains
   (a plot, a phenomenon, a question) before the formalism, mirroring principle
   6.

5. **Point to the notebook for the active step.** Notes state the idea; the
   notebook is where the student does it. End each note section with an explicit
   "-> see notebook NN" hook rather than duplicating runnable content.

6. **No redundancy across notes and notebooks.** An equation lives in one
   canonical place and is referenced from the other, not copy-pasted. If the
   equation changes, it changes once. (Equation code itself is immutable without
   sign-off - update the note first, then the implementation.)

7. **Progressive disclosure and consistent vocabulary** carry over unchanged:
   one new concept at a time, terms reused across the whole set.

Keep notes in the repo alongside the notebooks so notation drift is caught in
the same review. Same review checklist habits apply: one-sentence takeaway per
section, shared notation, reminder-not-derivation, pointer to the notebook.

### Exercises in the notes

Single source, one toggle. Exercises live inline in the `.tex`; solutions sit
right beside them but are hidden by default. The `\ifsolutions` machinery is in
`math_preamble.tex`; do not hand-flip it.

```latex
\begin{exercise}
Show that as $\sigma^2 \to 0$ the posterior mean approaches the least-squares
projection.
\end{exercise}
\begin{solution}
As $\sigma^2 \to 0$, $M \to C\trp C$, so $M^{-1}C\trp\vy \to (C\trp C)^{-1}C\trp\vy$.
\end{solution}

\begin{exercise}[optional stretch]   % harder tier, skippable
...
\end{exercise}
```

Rules, mirroring the notebook exercises:

- **Inline, one idea each**, placed right where the concept is developed - not
  batched at the end. The standalone `kalman_filter_exercise.tex` worksheet is
  the exception, not the model.
- **Solutions always present, hidden by default.** Every `exercise` gets a
  `solution` right after it.
- **Harder exercises use `[optional stretch]`.** The non-stretch exercises alone
  must cover the section's active step.

Build both PDFs from the one source (`cd lectures`):

```
make student     # latent_dynamics_notes.pdf            - solutions hidden
make solutions   # latent_dynamics_notes_solutions.pdf  - solutions shown
make             # both
```

PDFs are gitignored. **Compile and commit both PDFs only at a version
release**, not on every edit.

## Review checklist

Before merging a notebook, confirm:

- [ ] Takeaway statable in one sentence.
- [ ] Runs top-to-bottom clean, no manual edits, Colab badge works.
- [ ] Every key equation has adjacent, name-matched implementation.
- [ ] No visible cell over ~15 lines that is not the point (else -> `code_pack`).
- [ ] Every concept has at least one inline micro-exercise (non-stretch).
- [ ] Harder exercises marked `> **Stretch (optional):**`.
- [ ] Each exercise has a collapsed `<details>` solution.
- [ ] Only fill-in cells error before solving; predict/tweak stay runnable.
- [ ] No plumbing duplicated across notebooks that a named helper could hold.
- [ ] Ends with a "you can now..." + transfer prompt.
- [ ] Vocabulary/API consistent with the rest of the sequence.
- [ ] `.py` and `.ipynb` both committed and in sync.
