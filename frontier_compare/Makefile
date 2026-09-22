PY ?= .venv/bin/python  # relative to repo root (targets cd ..)

all: parse evaluate figures secondary

parse:
	cd .. && $(PY) frontier_compare/03_parse_responses.py

evaluate:
	cd .. && $(PY) frontier_compare/04_evaluate.py $(EVAL_ARGS)

figures:
	cd .. && $(PY) frontier_compare/05_figure4.py && $(PY) frontier_compare/05_figure4.py --aggregation pooled

secondary:
	cd .. && $(PY) frontier_compare/06_secondary.py

.PHONY: all parse evaluate figures secondary
