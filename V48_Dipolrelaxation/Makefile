# Lists and Configurables
PLOTS = cleaned_data-set1.pdf fit_approx_set1.pdf integrated-fit-set1.pdf \
		cleaned_data-set2.pdf fit_approx_set2.pdf integrated-fit-set2.pdf

TEXOPTIONS = -lualatex \
			 --output-directory=build \
			 --interaction=nonstopmode \
			 --halt-on-error

TEXHEADERS = ../head.tex \
			 ../math-commands.tex

#techcontent
TEXCONTENT = 00_main.tex \
			 01_theorie.tex  \
			 02_auswertung.tex


# Dont touch this laalaaalalaaaa
PLOTS_WITH_PATH = $(patsubst %,build/plots/%,$(PLOTS))



all: build/00_main.pdf

build/00_main.pdf: $(TEXCONTENT) $(TEXHEADERS) $(PLOTS_WITH_PATH) | build
	latexmk $(TEXOPTIONS) build 00_main.tex

$(PLOTS_WITH_PATH): data/ana.py data/set*.txt | build
	cd data; \
	./ana.py

build:
	mkdir -p build/plots build/tex

clean:
	rm -r build
