TEXOPTIONS = -lualatex \
			 --output-directory=build \
			 --interaction=nonstopmode \
			 --halt-on-error

TEXCONTENT = 02_auswertung.tex \
			 01_theorie.tex \
			 03_diskussion.tex

TEXHEADERS = ../head.tex \
			 ../math-commands.tex

all: build/00_main.pdf

build/00_main.pdf: 00_main.tex $(TEXHEADERS) 00_main.tex $(TEXCONTENT) | build
	latexmk $(TEXOPTIONS) 00_main.tex

build:
	mkdir -p build

clean:
	rm -r build

fast:
	lualatex $(TEXOPTIONS) --draftmode 00_main.tex
