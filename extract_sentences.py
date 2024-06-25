import pandas as pd
import os
import numpy as np

# Define the column names
columns = ['PMID', 'question', 'Sentences']

# Initialize an empty DataFrame with the specified columns
result_df = pd.DataFrame(columns=columns)

for filename in os.listdir("excel"):
    if filename.endswith('.xlsx'):
        excel_path = os.path.join("excel", filename)
        pmid = filename.split("_")[0]
        df = pd.read_excel(excel_path)
        for index, row in df.iterrows():
            question = row["question"]
            
            sentences = row["Sentences to keep (for sure)"]
            
            
            if type(sentences) == str:
                sentences = sentences.split("\"")
                sentences = [s.strip() for s in sentences]
                filtered_sentences = [s for s in sentences if len(s) > 25]
                print("Sentences")
                print(filtered_sentences)
                for sentence in filtered_sentences:
                    tmp_row = {'PMID': pmid, 'question': question, 'Sentences': sentence}
                    # result_df = result_df.append(tmp_row, ignore_index=True)
                    result_df = pd.concat([result_df, pd.DataFrame([tmp_row])], ignore_index=True)
result_df.to_excel('converted_labels.xlsx', index=False)