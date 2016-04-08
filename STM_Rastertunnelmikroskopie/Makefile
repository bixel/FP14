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

all: build/00_main.pdf

build/00_main.pdf: $(TEXCONTENT) $(TEXHEADERS) build/.ana-done | build
	latexmk $(TEXOPTIONS) 00_main.tex

analysis: raw/*.txt | build
	jupyter notebook analysis.ipynb

build/.ana-done: raw/*.txt build/analysis.py | build
	ipython3 build/analysis.py
	touch build/.ana-done

build/analysis.py: analysis.ipynb | build
	jupyter nbconvert --to=script --output=build/analysis analysis.ipynb

build:
	mkdir -p build/plots build/tex

clean:
	rm -rf build
