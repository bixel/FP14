all: build/00_main.pdf

build/00_main.pdf: 00_main.tex 01_theorie.tex 02_measurement.tex build/plots/eu_dist.pdf build/plots/calibration.pdf build/plots/comparison.pdf build/plots/efficiency.pdf build/plots/caesium_gauss.pdf build/plots/ba_activity.pdf pic/*.png | build
	lualatex --interaction=nonstopmode --halt-on-error --output-directory=build 00_main.tex

bib: build/00_main.pdf
	biber ./build/00_main && \
	lualatex --interaction=nonstopmode --halt-on-error --output-directory=build 00_main.tex && \
	lualatex --interaction=nonstopmode --halt-on-error --output-directory=build 00_main.tex

build/plots/eu_dist.pdf build/plots/calibration.pdf build/data/calibration.txt build/data/calibration-maxima.txt: analysis/calibration.py analysis/matplotlibrc analysis/peakdetect.py data/Europium.txt data/eu_spectrum_lit.txt | build
	cd analysis; \
	python3 calibration.py

build/plots/comparison.pdf: analysis/eu_compare.py data/Europium.txt data/Europium_proof.txt | build
	cd analysis; \
	python3 eu_compare.py

build/plots/efficiency.pdf: analysis/efficiency.py build/data/calibration-maxima.txt build/data/calibration.txt data/eu_spectrum_lit.txt | build
	cd analysis; \
	python3 efficiency.py

build/plots/caesium_gauss.pdf: analysis/caesium.py build/data/calibration.txt data/Caesium.txt | build
	cd analysis; \
	python3 caesium.py

build/plots/ba_activity.pdf build/data/ba_activity_peaks.txt build/data/ba_activity-params.txt build/data/ba_activity.txt: analysis/baactivity.py data/ba_spectrum_lit.txt build/data/efficiency-params.txt build/data/calibration.txt data/Barium.txt
	cd analysis; \
	python3 baactivity.py

build:
	mkdir -p build build/plots build/data

clean:
	rm -r ./build
