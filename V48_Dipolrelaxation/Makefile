# Lists and Configurables
PLOTS = data.pdf fit_approx_set1.pdf fit_approx_set2.pdf


# Dont touch this laalaaalalaaaa
PLOTS_WITH_PATH = $(patsubst %,build/plots/%,$(PLOTS))

all: build/00_main.pdf

build/00_main.pdf: 00_main.tex ../head.tex ../math-commands.tex \
		../00_lit.bib $(PLOTS_WITH_PATH) | build
	lualatex --output-directory build 00_main.tex

$(PLOTS_WITH_PATH): data/ana.py data/set*.txt | build
	cd data; \
	./ana.py

bib: build/00_main.pdf
	biber build/00_main; \
	lualatex --output-directory build 00_main.tex

build:
	mkdir -p build/plots

clean:
	rm -r build
