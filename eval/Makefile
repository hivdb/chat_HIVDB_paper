PYTHON ?= ../advanced-prompting/.venv/bin/python
PYTHONPATH ?= ..
export PYTHONPATH

REPO_ROOT := $(abspath ..)

.PHONY: all datasets evaluation evaluation_new30 evaluation_full150 evaluation_original120 learning_curve learning_curve_new30 learning_curve_full150 plot_learning_curve plot_learning_curve_new30 plot_learning_curve_full150 failure_prev analysis_by_qid error_types tests

MERGED_NEW30 = $(REPO_ROOT)/advanced-prompting/csv/merged_answers_new30.xlsx
MERGED_FULL150 = $(REPO_ROOT)/advanced-prompting/csv/merged_answers_full_150.xlsx
MERGED_ORIGINAL120 = $(REPO_ROOT)/advanced-prompting/csv/merged_answers_original_120.xlsx

LC_RESP_SIZE050_NEW30 = $(REPO_ROOT)/eval/learning-curve/responses/size050_new30_responses.csv
LC_RESP_SIZE100_NEW30 = $(REPO_ROOT)/eval/learning-curve/responses/size100_new30_responses.csv
LC_RESP_SIZE150_NEW30 = $(REPO_ROOT)/eval/learning-curve/responses/size150_new30_responses.csv
LC_RESP_SIZE200_NEW30 = $(REPO_ROOT)/eval/learning-curve/responses/size200_new30_responses.csv
LC_RESP_FT_NEW30 = $(REPO_ROOT)/eval/learning-curve/responses/ft_new30_responses.csv
LC_RESP_BASE_NEW30 = $(REPO_ROOT)/eval/learning-curve/responses/base_new30_responses.csv

LC_RESP_SIZE050_FULL150 = $(REPO_ROOT)/eval/learning-curve/responses/size050_full150_responses.csv
LC_RESP_SIZE100_FULL150 = $(REPO_ROOT)/eval/learning-curve/responses/size100_full150_responses.csv
LC_RESP_SIZE150_FULL150 = $(REPO_ROOT)/eval/learning-curve/responses/size150_full150_responses.csv
LC_RESP_SIZE200_FULL150 = $(REPO_ROOT)/eval/learning-curve/responses/size200_full150_responses.csv

all: datasets evaluation_full150 evaluation_new30 evaluation_original120 learning_curve_full150 learning_curve_new30 plot_learning_curve_full150 plot_learning_curve_new30 failure_prev analysis_by_qid error_types tests

datasets:
	$(PYTHON) build_datasets.py

evaluation: datasets
	$(PYTHON) evaluation.py

evaluation_new30: datasets
	$(PYTHON) evaluation.py --merged-path $(MERGED_NEW30) --output-suffix new30

evaluation_full150: datasets
	$(PYTHON) evaluation.py --merged-path $(MERGED_FULL150) --output-suffix full150

evaluation_original120: datasets
	$(PYTHON) evaluation.py --merged-path $(MERGED_ORIGINAL120) --output-suffix original120

learning_curve: datasets
	$(PYTHON) learning-curve/06_evaluate_learning_curve.py

learning_curve_new30: datasets
	$(PYTHON) learning-curve/06_evaluate_learning_curve.py --merged-path $(MERGED_NEW30) --output-suffix new30 \
		--responses size050=$(LC_RESP_SIZE050_NEW30) \
		--responses size100=$(LC_RESP_SIZE100_NEW30) \
		--responses size150=$(LC_RESP_SIZE150_NEW30) \
		--responses size200=$(LC_RESP_SIZE200_NEW30) \
		--responses ft=$(LC_RESP_FT_NEW30) \
		--responses base=$(LC_RESP_BASE_NEW30)

learning_curve_full150: datasets
	$(PYTHON) learning-curve/06_evaluate_learning_curve.py --merged-path $(MERGED_FULL150) --output-suffix full150 \
		--responses size050=$(LC_RESP_SIZE050_FULL150) \
		--responses size100=$(LC_RESP_SIZE100_FULL150) \
		--responses size150=$(LC_RESP_SIZE150_FULL150) \
		--responses size200=$(LC_RESP_SIZE200_FULL150)

plot_learning_curve: learning_curve
	$(PYTHON) learning-curve/07_plot_learning_curve.py

plot_learning_curve_new30: learning_curve_new30 evaluation_new30
	$(PYTHON) learning-curve/07_plot_learning_curve.py \
		--metrics $(REPO_ROOT)/eval/learning-curve/results/learning_curve_metrics_new30.csv \
		--base-results $(REPO_ROOT)/eval/results/evaluation_metrics_new30.csv \
		--significance $(REPO_ROOT)/eval/learning-curve/results/learning_curve_significance_new30.json \
		--suffix new30 --title "Learning Curve Analysis (new30)"

plot_learning_curve_full150: learning_curve_full150 evaluation_full150
	$(PYTHON) learning-curve/07_plot_learning_curve.py \
		--metrics $(REPO_ROOT)/eval/learning-curve/results/learning_curve_metrics_full150.csv \
		--base-results $(REPO_ROOT)/eval/results/evaluation_metrics_full150.csv \
		--significance $(REPO_ROOT)/eval/learning-curve/results/learning_curve_significance_full150.json \
		--suffix full150 --title "Learning Curve Analysis (full 150)"

failure_prev:
	$(PYTHON) error_analysis/failure_prev.py

analysis_by_qid:
	$(PYTHON) error_analysis/analysis_by_qid.py

error_types: analysis_by_qid
	$(PYTHON) error_analysis/error_types.py

tests: evaluation_full150 evaluation_new30 evaluation_original120
	$(PYTHON) -m eval.tests
