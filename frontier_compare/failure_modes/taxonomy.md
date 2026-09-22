# Failure-mode taxonomy

Used to label rows in `labeling_sheet.csv` (one row per frontier-model error). Assign exactly one
primary `failure_mode`. Set `annotation_error = yes` when, after checking the PDF, the model is
right and the human annotation is wrong or ambiguous. Those rows are reported separately and
never silently rescored.

The `Outcome` column (FP / FN_missed / FN_wrong_value) is assigned automatically by the scorer.
`failure_mode` is the curator's judgment of *why* the answer was wrong.

## Known modes (from the paper's error analysis, `eval/error_analysis/error_analysis.md`)

| Code | Mode | Description | Typical QIDs |
|---|---|---|---|
| `ABSTAIN` | Over-abstention | Evidence is in the paper but the model says "Not reported" / "None" | 5, 6, 9, 10, 11 |
| `SCOPE` | Wrong subpopulation | Answer describes a different group than the sequenced one (e.g. all enrolled vs. sequenced; cohort vs. subset with failure) | 5, 6, 7, 14–16 |
| `LAB_VS_CLINICAL` | Lab vs clinical confusion | Treats site-directed mutants / lab strains as patient sequences, or the reverse | 1, 8, 9 |
| `CONFLATE` | Concept conflation | Mixes related concepts: drug class vs drug, cloning vs SGS/NGS, genotypic vs phenotypic, ART-experienced vs virological failure | 2, 8, 12, 14, 15, 16 |
| `LIST_INCOMPLETE` | Incomplete list | Correct direction but missing items beyond the partial-match tolerance | 4, 6, 9, 15, 16 |
| `LIST_EXTRA` | Over-inclusive list | Adds items not supported for the sequenced population | 6, 15, 16 |
| `NUMERIC` | Wrong count or range | Wrong denominator, off-by-subset, or years of publication instead of sampling | 5, 7 |
| `FORMAT` | Format or normalization | Semantically right but the scorer can't match it (synonym, abbreviation, verbosity) | any |

## New modes to watch for with full-text PDFs

| Code | Mode | Description |
|---|---|---|
| `MISSED_TABLE_FIG` | Missed table/figure evidence | The answer is in a table, figure, or legend and the model missed or misread it (check `EvidenceLocation`) |
| `UNSUPPORTED` | Unsupported inference | Answer is plausible from background knowledge but not stated in the paper (e.g. inferring a country from author affiliation, or regimens from guidelines) |
| `FABRICATED_EVIDENCE` | Fabricated quote | Quoted evidence does not appear in the PDF |
| `SUPPLEMENT` | Supplement-only evidence | The human answer depends on supplementary material not in the PDF; not a model error in the strict sense |
| `OTHER` | Other | Describe in `notes` |

## Scientific-significance flag

For each error, the report separately tallies whether it would corrupt an HIVDB record
(`SCOPE`, `LAB_VS_CLINICAL`, `UNSUPPORTED`, `FABRICATED_EVIDENCE`, and wrong accessions, drugs,
or counts) versus errors a curator would catch trivially or that don't change the record
(`FORMAT`, `SUPPLEMENT`, annotation errors).
