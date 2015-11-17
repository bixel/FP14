# Lists and Configurables
PLOTS = data.pdf fit_approx_set1.pdf fit_approx_set2.pdf\
		integrated-fit-set1.pdf integrated-fit-set2.pdf


# Dont touch this laalaaalalaaaa
PLOTS_WITH_PATH = $(patsubst %,build/plots/%,$(PLOTS))

#techcontent
TEXCONTENT = 00_main.tex \
				01_theorie.tex  \
				02_auswertung.tex



all: build/00_main.pdf

build/00_main.pdf: $(TEXCONTENT) ../head.tex ../math-commands.tex \
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
