TEXOPTIONS = -lualatex \
			 --output-directory=build \
			 --interaction=nonstopmode \
			 --halt-on-error

TEXHEADERS = ../head.tex \
			 ../math-commands.tex

TEXCONTENT = 01_theorie.tex \
			 02_auswertung.tex \
			 03_diskussion.tex

PLOTS = build/Cv*.pdf \
		build/Theta.tex \
		build/omega_D_theo.tex \
		build/Fits.tex \
		build/theta_D.tex \
		build/theta_D_T.tex \
		build/theta_D_theo.tex \
		build/data.tex

all: build/00_main.pdf

build/00_main.pdf: 00_main.tex $(TEXHEADERS) 00_main.tex $(TEXCONTENT) $(PLOTS) | build
	latexmk $(TEXOPTIONS) 00_main.tex

$(PLOTS): Cv.py data/daten.txt data/temp.txt | build
	python3 Cv.py

build:
	mkdir -p build

clean:
	rm -r build

cache:
	rm -rf C:\Users\Markud\AppData\Local\Temp\par-4d61726b7564\cache-c38a9b05b97a81d339b4947ba00f0aacefe00497

fast:
	lualatex $(TEXOPTIONS) 00_main.tex