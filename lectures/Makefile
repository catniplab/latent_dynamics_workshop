# Self-contained build: catniplab.sty and math_preamble.tex live in this
# directory, so no TEXINPUTS setup is needed. Just run `make` from lectures/.
#
# Two outputs from one source (see the exercise/solution toggle in
# math_preamble.tex):
#   make            -> both PDFs
#   make student    -> latent_dynamics_notes.pdf            (solutions hidden)
#   make solutions  -> latent_dynamics_notes_solutions.pdf  (solutions shown)
# PDFs are gitignored; compile and commit them only at a version release.

DOC = latent_dynamics_notes
DEPS = $(DOC).tex references.bib catniplab.sty math_preamble.tex

all: student solutions

student: $(DOC).pdf

$(DOC).pdf: $(DEPS)
	latexmk -pdf -interaction=nonstopmode $(DOC).tex

# Instructor build: prepend \def\SolutionsBuild{} before \input, and use a
# distinct jobname so the two builds do not share aux/pdf files.
solutions: $(DOC)_solutions.pdf

$(DOC)_solutions.pdf: $(DEPS)
	latexmk -pdf -interaction=nonstopmode -jobname=$(DOC)_solutions \
	    -pdflatex='pdflatex -interaction=nonstopmode %O "\def\SolutionsBuild{}\input{%S}"' \
	    $(DOC).tex

clean:
	latexmk -c
	latexmk -c -jobname=$(DOC)_solutions

.PHONY: all student solutions clean
