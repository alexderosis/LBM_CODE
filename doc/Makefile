# Build the scheme/validation document.
#   make          -> m3lb.pdf
#   make figures  -> regenerate the figure data from scratch (needs the solver built)
#   make clean    -> remove auxiliary files
#
# The figure inputs are committed, so `make` alone works without running anything.

m3lb.pdf: m3lb.tex $(wildcard fig/*.dat) $(wildcard fig/*.png)
	pdflatex -interaction=nonstopmode $< >/dev/null
	pdflatex -interaction=nonstopmode $< >/dev/null
	@rm -f m3lb.aux m3lb.log m3lb.out m3lb.toc
	@echo "wrote $@"

# Convergence data: ../build/validation/grid_convergence > fig/grid.csv  (~45 s)
# Orszag-Tang snapshots: run in fig/ so the .bin files land there  (~2 min)
figures:
	cd fig && ../../build/validation/grid_convergence > grid.csv
	cd fig && ../../build/validation/orszag_tang -n 512 -op cm -dump
	cd fig && python3 mkpng.py && rm -f *.bin
	@echo "figure inputs regenerated; now run make"

clean:
	rm -f m3lb.aux m3lb.log m3lb.out m3lb.toc
.PHONY: clean figures
