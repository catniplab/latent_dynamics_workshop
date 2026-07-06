# Self-contained build: catniplab.sty and math_preamble.tex live in this
# directory, so no TEXINPUTS setup is needed. Just run `make` from lectures/.

DOC = latent_dynamics_notes

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex references.bib catniplab.sty math_preamble.tex
	latexmk -pdf -interaction=nonstopmode $(DOC).tex

clean:
	latexmk -c

.PHONY: all clean
