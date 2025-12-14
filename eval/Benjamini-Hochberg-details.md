# Benjamini–Hochberg (BH) adjustment in `eval/`

This summarizes exactly how BH is applied in the evaluation pipeline and shows concrete examples from the current `full150` outputs.

## Algorithm (what we do)

For any slice of p-values we need to adjust:
1. Sort p-values ascending and assign ranks 1..n.
2. Compute `adj = p * n / rank`.
3. Enforce monotonicity by taking a reverse cumulative minimum over the sorted `adj`.
4. Clip to [0, 1] and unsort back to the original order.

This is implemented in `eval/statistics.py::benjamini_hochberg`.

## Where it is applied and how many comparisons

- **Paired tests (t-test, Wilcoxon)**  
  For each metric (`accuracy`, `precision`, `recall`, `f1`) and each test type, we adjust across **9** comparisons: 3 families (GPT-4o, Llama3.1-70B, Llama3.1-8B) × 3 targets (FT, FT+QSP, QSP) vs their family base. BH runs on those 9 p-values per `(metric, test)` slice.

- **Fisher per-QID tests**  
  For each `(family, target, metric)` combination, BH is applied across **16** per-QID p-values (one per QID in the `full150` set). Adjusted values land in `adj_p_qid_*`.

- **Aggregated Fisher summary**  
  After summing counts across QIDs, BH is run per metric across the family/target rows to produce `adj_p_value` in `evaluation_metrics_fisher*.xlsx`.

- **Learning-curve outputs**  
  They reuse the same functions, so the same per-metric/per-test slicing applies as above.

## Worked examples (from `eval/results/statistical_tests_full150.xlsx`)

### Example A: Paired Wilcoxon, metric = precision

Raw p-values across the 9 comparisons (family base vs FT, FT+QSP, QSP):
```
GPT-4o   vs FT       p=0.001
GPT-4o   vs FT+QSP   p=0.024
GPT-4o   vs QSP      p=0.430
L70B     vs FT       p=0.009
L70B     vs FT+QSP   p=0.009
L70B     vs QSP      p=0.002
L8B      vs FT       p=0.036
L8B      vs FT+QSP   p=0.890
L8B      vs QSP      p=0.310
```

BH steps on these 9 p-values give adjusted values (rounded here):
```
0.004, 0.054, 0.540, 0.027, 0.027, 0.009, 0.081, 0.890, 0.465
```
These appear in the `adj_p` column of the “Paired Tests” sheet.

### Example B: Fisher per-QID, GPT-4o vs FT, metric = recall

There are 16 QIDs; raw p-values per QID include (selected):
```
QID 6: p=0.010
QID 9: p=0.011
QID 12: p=0.039
QID 16: p=0.129
... (remaining QIDs)
```

BH across the 16 p-values yields adjusted values (selected):
```
QID 6:  adj_p≈0.042
QID 9:  adj_p≈0.044
QID 12: adj_p≈0.117
QID 16: adj_p≈0.235
```
These are stored in the `adj_p_qid_*` columns and surfaced in the “Fisher Exact Test” sheet.

### Example C: Aggregated Fisher summary (counts summed over QIDs)

After summing TP/FP/TN/FN across all QIDs, we get one p-value per `(family, target, metric)`. BH is then applied per metric across those rows. Example for `recall`:
```
Raw p-values (families × targets):
GPT-4o FT:      0.022
GPT-4o FT+QSP:  0.006
GPT-4o QSP:     0.470
L70B FT:        0.650
L70B FT+QSP:    0.018
L70B QSP:       0.018
L8B FT:         0.600
L8B FT+QSP:     0.750
L8B QSP:        0.004
```

After BH (per metric, n=9 for recall):
```
GPT-4o FT:      adj_p≈0.059
GPT-4o FT+QSP:  adj_p≈0.018
GPT-4o QSP:     adj_p≈0.470
L70B FT:        adj_p≈0.731
L70B FT+QSP:    adj_p≈0.041
L70B QSP:       adj_p≈0.041
L8B FT:         adj_p≈0.731
L8B FT+QSP:     adj_p≈0.750
L8B QSP:        adj_p≈0.018
```
These adjusted values appear in `evaluation_metrics_fisher_full150.xlsx` under `adj_p_rec_fisher`.
