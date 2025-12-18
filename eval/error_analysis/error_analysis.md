# Representative failure cases (24 / 2400 rows where all 12 models were wrong)

Scope: all-model-wrong rows from `eval/results/detailed_evaluation_full150.xlsx` (150 papers × 16 questions). For each case we note the human answer, what evidence is (not) in the corresponding checked paper (`advanced-prompting/papers/{PMID}/{PMID}.checked.md`), and the most common model answers (aggregated across the 12 model variants).

| # | PMID | QID (type) | Question (short) | Human answer | Evidence in paper? | Typical model answers |
| - | ---- | ---------- | ---------------- | ------------ | ------------------ | --------------------- |
| 1 | 28559249 | 5 (Number) | How many individuals were sequenced? | 1 | ✅ (explicit) | `0`, `Not reported` |
| 2 | 35913500 | 6 (List) | Countries of sequenced samples? | US | ✅ (appears) | `No`, `Not applicable`, `Not reported` |
| 3 | 35913500 | 9 (List) | Which HIV genes were sequenced? | RT, IN | ✅ (RT/IN mentioned) | `No`, `Not applicable`, `Not reported` |
| 4 | 35945163 | 5 (Number) | How many individuals were sequenced? | 6 | ✅ (explicit) | `0`, `35`, `Not reported`, `Not applicable` |
| 5 | 36659824 | 5 (Number) | How many individuals were sequenced? | 14 | ✅ (explicit) | `0`, `112`, `118` |
| 6 | 36660819 | 5 (Number) | How many individuals were sequenced? | 92 | ✅ (explicit) | `0`, `1045`, `Not reported` |
| 7 | 36694270 | 16 (List) | Drugs received before sequencing? | 3TC, TDF, EFV, LPV/r, EVG, ABC, DTG | ⚠️ Paper only shows 3TC, TDF, EFV, LPV/r; missing EVG/ABC/DTG | `3TC; TDF`, `No`, `Not reported`, truncated regimens |
| 8 | 36961945 | 6 (List) | Countries of sequenced samples? | USA | ⚠️ USA string not found in checked paper | `No`, `Not reported` |
| 9 | 37376649 | 11 (List) | What type of samples were sequenced? | PBMC | ⚠️ “PBMC” absent; blood terms present | `Whole blood`, `PBMC`, `No` |
| 10 | 37554471 | 8 (Boolean) | Were samples cloned? | No | ✅ | `No`, `Not reported` (graded wrong because of ambiguity) |
| 11 | 37701387 | 8 (Boolean) | Were samples cloned? | No | ✅ | `No`, `Not reported` (graded wrong because of uncertainty) |
| 12 | 37775947 | 6 (List) | Countries of sequenced samples? | Asia | ⚠️ “Asia” string not in paper | `South America`, `Africa`, `USA`, `No`, `Not reported` |
| 13 | 37823653 | 9 (List) | Which HIV genes were sequenced? | NFLG | ⚠️ “NFLG” not in paper; Env/Envelope present | `Near full length genome`, `Env`, `Envelope` |
| 14 | 38058846 | 10 (List) | What method was used for sequencing? | Next-generation sequencing (NGS) | ✅ | `No`, `IN`, `Not reported` |
| 15 | 38090027 | 5 (Number) | How many individuals were sequenced? | 1040 | ✅ | `0`, `Not reported`, `1040` (but often flagged as “not reported”) |
| 16 | 40431710 | 10 (List) | What method was used for sequencing? | Sanger sequencing | ⚠️ “Sanger” not present; method unclear | `NGS`, `High-throughput sequencing`, `Not reported` |
| 17 | 40431710 | 16 (List) | Drugs received before sequencing? | AZT, ddI, d4T, 3TC, ABC, TDF, TAF, DLV, ETR, EFV, RPV, SQV, IDV, RTV, LPV/r, DRV, ATV/r, ETV | ⚠️ Long historical regimen list absent from paper | Mixed long ARV lists; `Not reported` |
| 18 | 40779404 | 16 (List) | Drugs received before sequencing? | TFV (no other ARVs specified) | ⚠️ TFV not found in checked paper | `TFV`, vague ARV lists, `Not reported` |
| 19 | 40872801 | 10 (List) | What method was used for sequencing? | Sanger sequencing | ⚠️ “Sanger” not present; method not specified | `Not specified`, `Sequencing performed`, `Not reported` |
| 20 | 41012586 | 11 (List) | What type of samples were sequenced? | Plasma | ⚠️ “Plasma” not explicit in checked text | `Not specified`, `Whole blood`, `Not reported` |
| 21 | 41057785 | 10 (List) | What method was used for sequencing? | Sanger sequencing | ⚠️ “Sanger” not present | `Not specified`, `Sequencing performed`, `Not reported` |
| 22 | 41091504 | 8 (Boolean) | Were samples cloned prior to sequencing? | Yes (site-directed mutants) | ⚠️ Statement not found in paper | `Not mentioned`, `No`, `Not reported` |
| 23 | 41130593 | 5 (Number) | How many individuals were sequenced? | 64 | ⚠️ “64” not in paper | `Not stated`, small cohorts, `Not reported` |
| 24 | 41130593 | 8 (Boolean) | Were samples cloned prior to sequencing? | Yes (single genome) | ⚠️ “single genome” cloning not in paper | `Not indicated`, `No`, `Not reported` |

Notes on evidence alignment:
- Seven cases (e.g., #7, #16–18, #21–24) likely reflect label–document mismatch: required drugs/methods/counts are not stated in the checked paper, making the human answer effectively unanswerable by text-only LLMs.
- Five geography/sample-type items (#8, #12, #20) also lack the exact strings from the human answers, suggesting either synonym drift or missing metadata.
- When the paper does contain the answer (e.g., counts in #1, #4–6, #15), models tend to fail by under-recall or by defaulting to “Not reported,” not by hallucinating extra entities.

Implications:
- Some graded “errors” are dataset issues (answer not recoverable from the provided paper); these should be flagged or re-labeled to avoid penalizing models for missing evidence.
- For recoverable cases, failures cluster on list aggregation (drug regimens, gene panels) and cautious defaults (“Not reported”), suggesting span-aggregation and stronger recall bias would help.***
