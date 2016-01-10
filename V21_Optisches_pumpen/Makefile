# Lists and Configurables
PLOTS = transmission.pdf

TEXOPTIONS = -lualatex \
			 --output-directory=build \
			 --interaction=nonstopmode \
			 --halt-on-error

TEXHEADERS = ../head.tex \
			 ../math-commands.tex

#techcontent
TEXCONTENT = 00_main.tex \
			 01_theorie.tex  \
			 02_auswertung.tex \


# Dont touch this laalaaalalaaaa
PLOTS_WITH_PATH = $(patsubst %,build/plots/%,$(PLOTS))


all: build/00_main.pdf

build/00_main.pdf: $(TEXCONTENT) $(TEXHEADERS) $(PLOTS_WITH_PATH) | build
	latexmk $(TEXOPTIONS) 00_main.tex

$(PLOTS_WITH_PATH): img/plot.py matplotlibrc | build
	./img/plot.py

build:
	mkdir -p build/plots build/tex

clean:
	rm -r build
