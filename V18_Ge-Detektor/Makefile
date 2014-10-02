all: build/00_main.pdf

build/00_main.pdf: 00_main.tex 01_theorie.tex 02_measurement.tex | build
	lualatex --interaction=nonstopmode --halt-on-error --output-directory build 00_main.tex

bib: build/00_main.pdf
	biber ./build/00_main && \
	lualatex --interaction=nonstopmode --halt-on-error --output-directory build 00_main.tex

01_theorie.tex: pic/*.png

02_measurement.tex: build/plots/eu_dist.pdf build/plots/calibration.pdf

build/plots/eu_dist.pdf build/plots/calibration.pdf build/data/calibration.txt build/data/calibration-maxima.txt: analysis/calibration.py analysis/matplotlibrc analysis/peakdetect.py data/Europium.txt data/eu_spectrum_lit.txt | build
	cd analysis; \
	python3 calibration.py

data/0*_*.pdf: data/*.txt data/coal.py data/hist.py data/halbwertsbreite.py
	cd data; \
	python3 hist.py; \
	python3 coal.py; \
	python3 halbwertsbreite.py

build:
	mkdir -p build build/plots build/data

clean:
	rm -r ./build; \
	rm data/0*_*.pdf
