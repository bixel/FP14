all: build/00_main.pdf

build/00_main.pdf: 00_main.tex 01_theorie.tex 02_measurement.tex | build
	lualatex --output-directory build 00_main.tex

bib: build/00_main.pdf
	bibtex ./build/00_main && \
	lualatex --output-directory build 00_main.tex

01_theorie.tex: pic/*.png

02_measurement.tex: data/0*_*.pdf

data/0*_*.pdf: data/*.txt data/coal.py data/hist.py data/halbwertsbreite.py
	cd data; \
	python3 hist.py; \
	python3 coal.py; \
	python3 halbwertsbreite.py

build:
	mkdir -p build

clean:
	rm -r ./build; \
	rm data/0*_*.pdf
