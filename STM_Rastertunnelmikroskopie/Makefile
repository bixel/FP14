# Lists and Configurables
PLOTS =

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

PLOTS_WITH_PATH = $(patsubst %,build/plots/%,$(PLOTS))

all: build/00_main.pdf

build/00_main.pdf: $(TEXCONTENT) $(TEXHEADERS) $(PLOTS_WITH_PATH) | build
	latexmk $(TEXOPTIONS) 00_main.tex

analysis: raw/*.txt | build
	jupyter notebook analysis.ipynb

build:
	mkdir -p build/{plots}

clean:
	rm -rf build
