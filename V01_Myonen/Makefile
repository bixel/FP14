00_main: 00_main.tex
	mkdir -p ./build
	lualatex --output-directory ./build $^

bib: 00_main.tex
	mkdir -p ./build
	lualatex --output-directory ./build $^
	bibtex ./build/00_main
	lualatex --output-directory ./build $^
	lualatex --output-directory ./build $^


clean:
	rm -rf ./build
