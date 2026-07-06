# Self-contained build: catniplab.sty and math_preamble.tex live in this
# directory, so no TEXINPUTS setup is needed. Just run `make` from lectures/.
#
# Targets:
#   make            -> all PDFs (notes student + solutions, Kalman worksheet)
#   make student    -> latent_dynamics_notes.pdf            (solutions hidden)
#   make solutions  -> latent_dynamics_notes_solutions.pdf  (solutions shown)
#   make exercise   -> kalman_filter_exercise.pdf           (print-and-fill worksheet)
# The notes carry the \ifsolutions toggle (see math_preamble.tex); the Kalman
# worksheet is a standalone fill-in sheet with no embedded solutions.
# PDFs are gitignored; compile and commit them only at a version release.

DOC = latent_dynamics_notes
EX  = kalman_filter_exercise
NOTES_DEPS = $(DOC).tex references.bib catniplab.sty math_preamble.tex
EX_DEPS    = $(EX).tex math_preamble.tex

all: student solutions exercise

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

clean:
	latexmk -c
	latexmk -c -jobname=$(DOC)_solutions
	latexmk -c $(EX).tex

.PHONY: all student solutions exercise clean
