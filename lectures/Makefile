# Self-contained build: catniplab.sty and math_preamble.tex live in this
# directory, so no TEXINPUTS setup is needed. Just run `make` from lectures/.
#
# Targets (student = solutions hidden, solutions = solutions shown):
#   make                      -> every PDF below
#   make student              -> latent_dynamics_notes.pdf
#   make solutions            -> latent_dynamics_notes_solutions.pdf
#   make exercise             -> kalman_filter_exercise.pdf            (blanks)
#   make exercise-solutions   -> kalman_filter_exercise_solutions.pdf  (answer key)
# Both documents share the \ifsolutions toggle (see exercise_setup.tex); the
# *_solutions builds prepend \def\SolutionsBuild{}. PDFs are gitignored; compile
# and commit them only at a version release.

DOC = latent_dynamics_notes
EX  = kalman_filter_exercise
SHARED  = exercise_setup.tex
NOTES_DEPS = $(DOC).tex references.bib catniplab.sty math_preamble.tex $(SHARED)
EX_DEPS    = $(EX).tex math_preamble.tex $(SHARED)

all: student solutions exercise exercise-solutions

student: $(DOC).pdf

$(DOC).pdf: $(NOTES_DEPS)
	latexmk -pdf -interaction=nonstopmode $(DOC).tex

# Instructor build: prepend \def\SolutionsBuild{} before \input, and use a
# distinct jobname so the two builds do not share aux/pdf files.
solutions: $(DOC)_solutions.pdf

$(DOC)_solutions.pdf: $(NOTES_DEPS)
	latexmk -pdf -interaction=nonstopmode -jobname=$(DOC)_solutions \
	    -pdflatex='pdflatex -interaction=nonstopmode %O "\def\SolutionsBuild{}\input{%S}"' \
	    $(DOC).tex

exercise: $(EX).pdf

$(EX).pdf: $(EX_DEPS)
	latexmk -pdf -interaction=nonstopmode $(EX).tex

exercise-solutions: $(EX)_solutions.pdf

$(EX)_solutions.pdf: $(EX_DEPS)
	latexmk -pdf -interaction=nonstopmode -jobname=$(EX)_solutions \
	    -pdflatex='pdflatex -interaction=nonstopmode %O "\def\SolutionsBuild{}\input{%S}"' \
	    $(EX).tex

clean:
	latexmk -c
	latexmk -c -jobname=$(DOC)_solutions
	latexmk -c $(EX).tex
	latexmk -c -jobname=$(EX)_solutions

.PHONY: all student solutions exercise exercise-solutions clean
