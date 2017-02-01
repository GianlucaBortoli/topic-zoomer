PROJECT=report
TEX=pdflatex
BUILDTEX=$(TEX) $(PROJECT).tex
#BIBTEX=bibtex

all:
	$(BUILDTEX)
	#$(BIBTEX) $(PROJECT)
	$(BUILDTEX)

clean:
	rm -f *.swp *.aux *.bbl *.blg *.log *.out *.toc *.pdf *~
