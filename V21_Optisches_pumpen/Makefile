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


TEX_C = build/tex/c_data_raw_0.tex \
	build/tex/c_data_raw_85.tex \
	build/tex/c_data_raw_87.tex \
	build/tex/c_data_b_0.tex \
	build/tex/c_data_b_87.tex \
	build/tex/c_data_b_85.tex \
	build/tex/c_data_erd.tex \
	build/tex/c_fit_m_87.tex \
	build/tex/c_fit_b_87.tex \
	build/tex/c_fit_m_85.tex \
	build/tex/c_fit_b_85.tex \
	build/tex/c_data_gi.tex \
	build/tex/c_ratio_87.tex \
	build/tex/c_ratio_85.tex \
	build/tex/c_data_ratio.tex \
	build/tex/c_I_87.tex \
	build/tex/c_I_85.tex \
	build/tex/c_vert.tex \
	build/tex/c_data_square_85_0.tex \
	build/tex/c_data_square_85_1.tex \
	build/tex/c_data_square_85_2.tex \
	build/tex/c_data_square_85_3.tex \
	build/tex/c_data_square_85_4.tex \
	build/tex/c_data_square_87_0.tex \
	build/tex/c_data_square_87_1.tex \
	build/tex/c_data_square_87_2.tex \
	build/tex/c_data_square_87_3.tex \
	build/tex/c_data_square_87_4.tex \
	build/tex/c_data_square_87.tex \
	build/tex/c_data_square_85.tex

PLOT_C = build/plots/c_fit_85.pdf \
		build/plots/c_fit_87.pdf

TEX_I = build/tex/t_85.tex \
	build/tex/t_87.tex \
	build/tex/i_average_87.tex \
	build/tex/i_average_85.tex \
	build/tex/i_fit_start_87.tex \
	build/tex/i_fit_start_85.tex \
	build/tex/i_fit_end_87.tex \
	build/tex/i_fit_end_85.tex \
	build/tex/i_exp_raw_87.tex \
	build/tex/i_exp_raw_85.tex \
	build/tex/i_fit_m_87.tex \
	build/tex/i_fit_b_87.tex \
	build/tex/i_fit_m_85.tex \
	build/tex/i_fit_b_85.tex \
	build/tex/i_fit_T_a_87.tex \
	build/tex/i_fit_T_a_85.tex \
	build/tex/i_fit_T_b_87.tex \
	build/tex/i_fit_T_b_85.tex \
	build/tex/i_fit_T_c_87.tex \
	build/tex/i_fit_T_c_85.tex \
	build/tex/i_fit_T_quot.tex


PLOT_I = build/plots/fit_L_87.pdf \
		build/plots/fit_L_85.pdf \
		build/plots/fit_E_87.pdf \
		build/plots/fit_E_85.pdf \
		build/plots/fit_T_87.pdf \
		build/plots/fit_T_85.pdf

TEX_OWN = c_magnet.tex \
		c_raw.tex \
		i_periode.tex

# Dont touch this laalaaalalaaaa

PLOTS_WITH_PATH = $(patsubst %,build/plots/%,$(PLOTS))

all: build/00_main.pdf

build/00_main.pdf: $(TEXCONTENT) $(TEXHEADERS) $(PLOTS_WITH_PATH) $(TEX_C) $(PLOT_C) $(TEX_I) $(PLOT_I) $(TEX_OWN)| build build/plots build/tex
	latexmk $(TEXOPTIONS) 00_main.tex

$(PLOTS_WITH_PATH): scripts/plot.py matplotlibrc | build build/plots build/tex
	python scripts/plot.py

$(TEX_C) $(PLOT_C): scripts/aufg_c.py | build build/plots build/tex
	python scripts/aufg_c.py

$(TEX_I) $(PLOT_I): scripts/aufg_i.py | build build/plots build/tex
	python scripts/aufg_i.py

# $(PLOTS_WITH_PATH): img/plot.py matplotlibrc | build
# 	./img/plot.py

build:
	mkdir -p build 

build/plots:
	mkdir -p build/plots

build/tex:
	mkdir -p build/tex

clean:
	rm -r build
