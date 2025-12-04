write a new python script to parse
csv/llama-3.1-70B-FT_new30.csv. 'FT Answer' column, save to a csv file with suffix '_parsed.csv'.

column headers are PMID, QID, Question, Answer.

Extract Question or QID or Both.
Extract Answer.

Find line starts with 'Question', then find in the next line any text contain the quesetion list below.

1. Does the paper report HIV sequences from patient samples?,
2. Does the paper report in vitro drug susceptibility data?,
3. Were sequences from the paper made publicly available?,
4. What were the GenBank accession numbers for sequenced HIV isolates?,
5. How many individuals had samples obtained for HIV sequencing?,
6. From which countries were the sequenced samples obtained?,
7. From what years were the sequenced samples obtained?,
8. Were samples cloned prior to sequencing?,
9. Which HIV genes were reported to have been sequenced?,
10. What method was used for sequencing?,
11. What type of samples were sequenced?,
12. Were any sequences obtained from individuals with virological failure on a treatment regimen?,
13. Were the patients in the study in a clinical trial?,
14. Does the paper report HIV sequences from individuals who had previously received ARV drugs?,
15. Which drug classes were received by individuals in the study before sample sequencing?,
16. Which drugs were received by individuals in the study before sample sequencing?,


If match, assign the number as QID, assign the question as 'Question'

find immediate behind it 'Answer' row assign as answer.

After that, use Table S4.xlsx to get QID by Question

Make sure each PMID has 16 QID/Question rows. If you cannot find leave the Answer in blank.
