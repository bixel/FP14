all:

analysis: raw/*.txt | build
	jupyter notebook analysis.ipynb

build:
	mkdir -p build/{plots}

clean:
	rm -rf build
