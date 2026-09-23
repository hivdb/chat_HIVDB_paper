PY ?= .venv/bin/python  # relative to repo root (targets cd ..)

all: parse evaluate figures secondary workbook

parse:
	cd .. && $(PY) frontier_compare/03_parse_responses.py

evaluate:
	cd .. && $(PY) frontier_compare/04_evaluate.py --allow-subset $(EVAL_ARGS)

figures:
	cd .. && $(PY) frontier_compare/05_figure4.py

secondary:
	cd .. && $(PY) frontier_compare/06_secondary.py

workbook:
	cd .. && $(PY) frontier_compare/07_detailed_evaluation.py

.PHONY: all parse evaluate figures secondary workbook
