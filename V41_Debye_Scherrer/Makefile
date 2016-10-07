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

build/00_main.pdf: $(TEXCONTENT) $(TEXHEADERS) build/.ref-done build/.ana-done  | build
	latexmk $(TEXOPTIONS) 00_main.tex

analysis: raw/*.dat| build
	jupyter notebook analysis.ipynb

build/.ana-done: build/analysis.py | build build/hkl_one build/hkl_two build/pic build/tex build/tex_tables
	ipython3 build/analysis.py
	touch build/.ana-done

build/analysis.py: analysis.ipynb | build build/hkl_one build/hkl_two build/pic build/tex build/tex_tables
	jupyter nbconvert --to=script --output=build/analysis analysis.ipynb

build/.ref-done: build/reflexes.py | build build/hkl_one build/hkl_two build/pic build/tex build/tex_tables
	ipython3 build/reflexes.py
	touch build/.ref-done

build/reflexes.py: reflexes.ipynb | build build/hkl_one build/hkl_two build/pic build/tex build/tex_tables
	jupyter nbconvert --to=script --output=build/reflexes reflexes.ipynb

build/hkl_one:
	mkdir -p build/hkl_one

build/hkl_two:
	mkdir -p build/hkl_two

build/pic:
	mkdir -p build/pic

build/tex:
	mkdir -p build/tex

build/tex_tables:
	mkdir -p build/tex_tables

build:
	mkdir -p build

clean:
	rm -rf build
