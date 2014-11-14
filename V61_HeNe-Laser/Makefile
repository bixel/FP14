all: build/polarisation_*.pdf build/TEM_*.pdf

build/polarisation_*.pdf: polarisation.py data/polarisation.txt | build
	python3 polarisation.py

build/TEM_*.pdf: TEM.py data/TEM_*.txt | build
	python3 TEM.py

build:
	mkdir -p build

clean:
	rm -r build
