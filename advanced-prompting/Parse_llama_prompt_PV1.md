write a new python script to parse
csv/llama-3.1-8B-PV1_new30_parsed.csv. 'Multiple Answer' column, save to a csv file with suffix '_parsed.csv'.

Extract Question or QID or Both.
Extract Answer.


column headers are PMID, QID, Question, Answer.

Parse Question from Question line like
"Question: From what years were the sequenced samples obtained"
It can be in other similar formats.
remove "Question: " and question id
save to Question, After that, use Table S4.xlsx to get QID.

Otherwise Extract QID in text like 'Question: 2', save to QID,

After that, use Table S4.xlsx to get Question.

Make sure each PMID has 16 QID/Question rows. If you cannot find leave the Answer in blank.

