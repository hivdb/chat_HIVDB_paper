PYTHON ?= ../.venv/bin/python
PYTHONPATH ?= ..
export PYTHONPATH

REPO_ROOT := $(abspath ..)

.PHONY: all datasets evaluation evaluation_new30 evaluation_full150 evaluation_original120 learning_curve learning_curve_new30 learning_curve_full150 plot_learning_curve plot_learning_curve_new30 plot_learning_curve_full150 failure_prev analysis_by_qid tests exact_vs_partial

MERGED_NEW30 = $(REPO_ROOT)/advanced-prompting/csv/merged_answers_new30.xlsx
MERGED_FULL150 = $(REPO_ROOT)/advanced-prompting/csv/merged_answers_full_150.xlsx
MERGED_ORIGINAL120 = $(REPO_ROOT)/advanced-prompting/csv/merged_answers_original_120.xlsx

LC_RESP_SIZE_ARGS_NEW30 = $(foreach s,050 100 150 200,--responses size$(s)=$(REPO_ROOT)/eval/learning-curve/responses/size$(s)_new30_responses.csv)
LC_RESP_FT_NEW30 = $(REPO_ROOT)/eval/learning-curve/responses/ft_new30_responses.csv
LC_RESP_BASE_NEW30 = $(REPO_ROOT)/eval/learning-curve/responses/base_new30_responses.csv
FT_SIZES := 50 100 150 200
LC_RESP_FTQSP_NEW30 = $(REPO_ROOT)/advanced-prompting/csv/gpt-4o/gpt-4o-mini-2024-07-18_FT_PV1_new30.xlsx
LC_RESP_FTQSP_ARGS_NEW30 = --responses ftqsp=$(LC_RESP_FTQSP_NEW30) $(foreach s,$(FT_SIZES),--responses ft$(s)=$(REPO_ROOT)/advanced-prompting/csv/gpt-4o/gpt-4o-mini-2024-07-18_FT-$(s)_PV1_new30.xlsx)

LC_RESP_SIZE_ARGS_FULL150 = $(foreach s,050 100 150 200,--responses size$(s)=$(REPO_ROOT)/eval/learning-curve/responses/size$(s)_full150_responses.csv)
LC_RESP_FTQSP_FULL150 = $(REPO_ROOT)/advanced-prompting/csv/gpt-4o/gpt-4o-mini-2024-07-18_FT_PV1_full150.xlsx
LC_RESP_FTQSP_ORIGINAL120 = $(REPO_ROOT)/advanced-prompting/csv/gpt-4o/gpt-4o-mini-2024-07-18_FT_PV1_original120.xlsx
LC_RESP_FTQSP_ARGS_FULL150 = --responses ftqsp=$(LC_RESP_FTQSP_FULL150) $(foreach s,$(FT_SIZES),--responses ft$(s)=$(REPO_ROOT)/advanced-prompting/csv/gpt-4o/gpt-4o-mini-2024-07-18_FT-$(s)_PV1_full150.xlsx)
LC_RESP_FTQSP_ARGS_ORIGINAL120 = --responses ftqsp=$(LC_RESP_FTQSP_ORIGINAL120) $(foreach s,$(FT_SIZES),--responses ft$(s)=$(REPO_ROOT)/advanced-prompting/csv/gpt-4o/gpt-4o-mini-2024-07-18_FT-$(s)_PV1_original120.xlsx)
LC_RESP_SIZE_ARGS_ORIGINAL120 = $(foreach s,050 100 150 200,--responses size$(s)=$(REPO_ROOT)/eval/learning-curve/archive/responses/size$(s)_responses.csv)

all: datasets evaluation_full150 evaluation_new30 evaluation_original120 learning_curve_full150 learning_curve_new30 learning_curve_original120 plot_learning_curve_full150 plot_learning_curve_new30 plot_learning_curve_original120 failure_prev analysis_by_qid exact_vs_partial tests

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
		$(LC_RESP_SIZE_ARGS_NEW30) \
		--responses ft=$(LC_RESP_FT_NEW30) \
		--responses base=$(LC_RESP_BASE_NEW30) \
		$(LC_RESP_FTQSP_ARGS_NEW30)

learning_curve_full150: datasets
	$(PYTHON) learning-curve/06_evaluate_learning_curve.py --merged-path $(MERGED_FULL150) --output-suffix full150 \
		$(LC_RESP_SIZE_ARGS_FULL150) \
        $(LC_RESP_FTQSP_ARGS_FULL150)

learning_curve_original120: datasets
	$(PYTHON) learning-curve/06_evaluate_learning_curve.py --merged-path $(MERGED_ORIGINAL120) --output-suffix original120 \
		$(LC_RESP_FTQSP_ARGS_ORIGINAL120) \
		$(LC_RESP_SIZE_ARGS_ORIGINAL120)

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
		--suffix full150 --title "GPT-4o: Learning Curve Analysis"

plot_learning_curve_original120: learning_curve_original120 evaluation_original120
	$(PYTHON) learning-curve/07_plot_learning_curve.py \
		--metrics $(REPO_ROOT)/eval/learning-curve/results/learning_curve_metrics_original120.csv \
		--base-results $(REPO_ROOT)/eval/results/evaluation_metrics_original120.csv \
		--significance $(REPO_ROOT)/eval/learning-curve/results/learning_curve_significance_original120.json \
		--suffix original120 --title "Learning Curve Analysis (original120)"

failure_prev:
	$(PYTHON) error_analysis/error_analysis.py failure_prev

analysis_by_qid:
	$(PYTHON) error_analysis/error_analysis.py analysis_by_qid


tests: evaluation_full150 evaluation_new30 evaluation_original120
	$(PYTHON) -m eval.tests

exact_vs_partial: datasets
	$(PYTHON) exact_vs_partial/list_metrics_by_scenario.py --merged-path $(MERGED_FULL150) --output-suffix full150
