all: build/00_main.pdf

build/00_main.pdf: 00_main.tex ../head.tex ../math-commands.tex ../00_lit.bib | build
	lualatex --output-directory build 00_main.tex

bib: build/00_main.pdf
	biber build/00_main; \
	lualatex --output-directory build 00_main.tex

build:
	mkdir -p build

clean:
	rm -r build
