TEXOPTIONS = -lualatex \
			 --output-directory=build \
			 --interaction=nonstopmode \
			 --halt-on-error

TEXCONTENT = 02_auswertung.tex \
			 01_theorie.tex \
			 03_diskussion.tex

IMAGES = images/am-diode-oberwellen.jpg \
		 images/am-diode-signal.png \
		 images/am-diode-spektrum.jpg \
		 images/am-signal.png \
		 images/am-spektrum.jpg \
		 images/am-amplitude.png \
		 images/modulationsschaltung.png \
		 images/ringmodulator.png
		 
TEXHEADERS = ../head.tex \
			 ../math-commands.tex

CALCS = build/m1.tex \
		build/m2.tex \
		build/t_t.tex \
		build/delta_f.tex \
		build/delta_t.tex

PLOTS = build/demodulation-cosinus.pdf

all: build/00_main.pdf

build/00_main.pdf: 00_main.tex $(TEXHEADERS) 00_main.tex $(TEXCONTENT) $(IMAGES) $(CALCS) $(PLOTS) | build
	latexmk $(TEXOPTIONS) 00_main.tex

$(CALCS) $(PLOTS): calculations.py am-demodulation-e.txt | build
	python calculations.py

build:
	mkdir -p build

clean:
	rm -r build

fast:
	lualatex $(TEXOPTIONS) --draftmode 00_main.tex
