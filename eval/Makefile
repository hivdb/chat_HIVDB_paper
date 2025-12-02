PYTHON ?= ../advanced-prompting/.venv/bin/python
PYTHONPATH ?= ..
export PYTHONPATH

.PHONY: all evaluation learning_curve plot_learning_curve failure_prev analysis_by_qid error_types

all: evaluation learning_curve plot_learning_curve failure_prev analysis_by_qid error_types

evaluation:
	$(PYTHON) evaluation.py

learning_curve:
	if ls learning-curve/responses/*_responses.csv >/dev/null 2>&1; then \
		$(PYTHON) learning-curve/06_evaluate_learning_curve.py; \
	else \
		echo "Skipping learning-curve evaluation: no responses found under learning-curve/responses/"; \
	fi

plot_learning_curve: learning_curve
	if [ -f learning-curve/results/learning_curve_metrics.csv ]; then \
		$(PYTHON) learning-curve/07_plot_learning_curve.py; \
	else \
		echo "Skipping learning-curve plotting: missing learning_curve_metrics.csv"; \
	fi

failure_prev:
	$(PYTHON) error_analysis/failure_prev.py

analysis_by_qid:
	$(PYTHON) error_analysis/analysis_by_qid.py

error_types: analysis_by_qid
	$(PYTHON) error_analysis/error_types.py
