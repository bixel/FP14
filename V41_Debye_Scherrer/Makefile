TEXOPTIONS = -lualatex \
			 --output-directory=build \
			 --interaction=nonstopmode \
			 --halt-on-error

TEXHEADERS = ../head.tex \
			 ../math-commands.tex \
			 ../00_lit.bib

#techcontent
TEXCONTENT = 00_main.tex \
			 01_theorie.tex  \
			 02_auswertung.tex \
			 03_anhang.tex

all: build/00_main.pdf

build/00_main.pdf: $(TEXCONTENT) $(TEXHEADERS) raw/*.dat | build
	latexmk $(TEXOPTIONS) 00_main.tex

analysis: raw/*.dat| build
	jupyter notebook analysis.ipynb

build:
	mkdir -p build/plots build/tex

clean:
	rm -rf build
