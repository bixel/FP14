TEXOPTIONS = -lualatex \
			 --output-directory=build \
			 --interaction=nonstopmode \
			 --halt-on-error

TEXHEADERS = ../head.tex \
			 ../math-commands.tex

TEXCONTENT = 01_theorie.tex \
			 02_auswertung.tex \
			 03_diskussion.tex

PLOTS = build/Cv_*.pdf 

all: build/00_main.pdf

build/00_main.pdf: 00_main.tex $(TEXHEADERS) 00_main.tex $(TEXCONTENT) $(PLOTS) | build
	latexmk $(TEXOPTIONS) 00_main.tex

build/Cv_*.pdf: Cv.py data/daten.txt data/temp.txt | build
	python3 Cv.py

build:
	mkdir -p build

clean:
	rm -r build
