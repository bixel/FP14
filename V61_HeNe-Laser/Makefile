TEXOPTIONS = --output-directory=build \
			 --interaction=nonstopmode \
			 --halt-on-error

TEXHEADERS = ../head.tex \
			 ../math-commands.tex

TEXCONTENT = 01_theorie.tex \
			 02_auswertung.tex \
			 build/tem00_*.tex

PLOTS = build/polarisation_*.pdf \
		build/TEM_*.pdf \
		img/laser.jpg

WAVELENGTH = build/lambdas_grid_cm.tex \
			 build/lambdas_grid_mm_l.tex \
			 build/lambdas_grid_mm_r.tex \
			 build/lambdas_slot_mm_l.tex \
			 build/lambdas_slot_mm_r.tex \
			 build/lambda_grid.tex \
			 build/lambda_slot.tex

all: build/00_main.pdf

build/00_main.pdf: build/00_main.bcf build/00_main.bbl
	lualatex $(TEXOPTIONS) 00_main.tex

build/00_main.bcf: $(TEXHEADERS) 00_main.tex $(TEXCONTENT) $(PLOTS) | build
	lualatex $(TEXOPTIONS) 00_main.tex && \
	lualatex $(TEXOPTIONS) 00_main.tex

build/00_main.bbl: build/00_main.bcf
	biber build/00_main.bcf

build/polarisation_*.pdf: polarisation.py data/polarisation.txt | build
	python3 polarisation.py

build/TEM_*.pdf build/tem00_*.tex: TEM.py data/TEM_*.txt | build
	python3 TEM.py

$(WAVELENGTH): wavelength.py data/grid*.txt data/slot.txt | build
	python3 wavelength.py

build:
	mkdir -p build

clean:
	rm -r build
