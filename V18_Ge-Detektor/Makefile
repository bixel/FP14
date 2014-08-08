00_main: 00_main.tex
	mkdir -p ./build
	lualatex --output-directory ./build $^

clean:
	rm -rf ./build
	./*.bbl
	./*.log
