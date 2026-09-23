PY ?= .venv/bin/python  # relative to repo root (targets cd ..)

all: parse evaluate figures secondary workbook adjudicate

parse:
	cd .. && $(PY) frontier_compare/03_parse_responses.py

evaluate:
	cd .. && $(PY) frontier_compare/04_evaluate.py $(EVAL_ARGS)

figures:
	cd .. && $(PY) frontier_compare/05_figure4.py

secondary:
	cd .. && $(PY) frontier_compare/06_secondary.py

workbook:
	cd .. && $(PY) frontier_compare/07_detailed_evaluation.py

adjudicate:
	cd .. && $(PY) frontier_compare/08_error_triage.py && $(PY) frontier_compare/11_auto_verdicts.py && $(PY) frontier_compare/10_adjudicate.py --apply

.PHONY: all parse evaluate figures secondary workbook adjudicate
