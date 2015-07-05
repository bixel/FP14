TEXOPTIONS = -lualatex \
			 --output-directory=build \
			 --interaction=nonstopmode \
			 --halt-on-error

TEXCONTENT = 02_auswertung.tex \
			 01_theorie.tex \
			 03_diskussion.tex

TEXHEADERS = ../head.tex \
			 ../math-commands.tex

PYTHON = part_a.py \
			part_c.py \
			part_de.py \
			part_h.py \
			part_i.py

all: build/00_main.pdf

build/00_main.pdf: 00_main.tex $(TEXHEADERS) 00_main.tex $(TEXCONTENT) build/*.tex | build
	latexmk $(TEXOPTIONS) 00_main.tex

build/*.tex: $(PYTHON) | build
	python $(PYTHON)

build:
	mkdir -p build

clean:
	rm -r build

fast:
	lualatex $(TEXOPTIONS) --draftmode 00_main.tex
