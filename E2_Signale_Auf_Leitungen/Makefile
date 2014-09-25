all: build/00_main.pdf

build/00_main.pdf: 00_main.tex 01_theorie.tex 02_auswertung.tex ../head.tex ../math-commands.tex ../00_lit.bib | build
	lualatex --output-directory build 00_main.tex

bib: build/00_main.pdf
	biber build/00_main; \
	lualatex --output-directory build 00_main.tex

02_auswertung.tex: data/a/R.pdf data/a/G.pdf data/a/L.pdf data/a/C.pdf \
data/b/alpha.pdf data/b/alpha_zeit.pdf \
data/c/Regression1.pdf data/c/Regression2.pdf \
data/d/Regression1.pdf data/d/Regression2.pdf data/d/Regression3.pdf

data/a/R.pdf data/a/G.pdf data/a/L.pdf data/a/C.pdf: data/a/*.txt data/a/plot_rlc.py
	cd data/a; \
	python plot_rlc.py

data/d/Regression1.pdf data/d/Regression2.pdf data/d/Regression3.pdf: data/d/abschluss.py
	cd data/d; \
	python abschluss.py
	
data/c/Regression1.pdf data/c/Regression2.pdf: data/c/*.py
	cd data/c; \
	python zeit.py; \
	python konst.py

build:
	mkdir -p build

clean:
	rm -r build
