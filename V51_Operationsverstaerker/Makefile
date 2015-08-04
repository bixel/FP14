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

build/00_main.pdf: 00_main.tex $(TEXHEADERS) 00_main.tex \
				$(TEXCONTENT) build/a_plot_1.pdf \
				build/c_plot.pdf build/d_plot.pdf \
				build/e_plot.pdf build/h_plot.pdf \
				build/i_exp_plot.pdf | build
	latexmk $(TEXOPTIONS) 00_main.tex

build/a_plot_1.pdf: part_a.py | build
	python part_a.py

build/c_plot.pdf: part_c.py | build
	python part_c.py

build/d_plot.pdf: part_de.py | build
	python part_de.py

build/h_plot.pdf: part_h.py | build
	python part_h.py

build/i_exp_plot.pdf: part_i.py | build
	python part_i.py

build:
	mkdir -p build

clean:
	rm -r build

fast:
	lualatex $(TEXOPTIONS) 00_main.tex
