from openai import OpenAI
import os
from dotenv import load_dotenv
from dotenv import find_dotenv
load_dotenv(find_dotenv())
import pandas as pd
import xlsxwriter

def query_gpt(system_prompt, paper_content, df, save_path):
    df['Answer'] = df['Answer'].astype(object)
    df['Reference Sentences'] = df['Answer'].astype(object)
    for i in range(len(df["question"])):
        message = [{"role": "system", "content": system_prompt}]
        formatted_user_prompt = user_prompt.format(paper_content=paper_content, question=df["question"][i], requirements= df["Prompt_additional"][i])
        message += [{"role": "user", "content": formatted_user_prompt}]

        # print(df.loc["question", i])
        # print(message)

        ## Set the API key and model name
        MODEL="gpt-4-turbo-2024-04-09"
        client = OpenAI(api_key=os.getenv('GPT-4O-KEY'))

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt}, # <-- This is the system message that provides context to the model
                {"role": "user", "content": formatted_user_prompt}  # <-- This is the user message for which the model will generate a response
            ],
            temperature=0
        )

        #print("Assistant: " + completion.choices[0].message.content)

        df.at[i, 'Answer'] = completion.choices[0].message.content.split("Reference:")[0]
        df.at[i, 'Reference Sentences'] = completion.choices[0].message.content.split("Reference:")[1]

    df.to_excel(save_path, index=False,engine='xlsxwriter') #to prevent encode errors use xlsxwriter
    return df


system_path = "prompt/system.txt"
system_prompt = ""
# Load the content of the text file into a string
with open(system_path, 'r', encoding='utf-8') as file:
    system_prompt = file.read()

user_path = "prompt/user.txt"
# Load the content of the text file into a string
with open(user_path, 'r', encoding='utf-8') as file:
    user_prompt = file.read()


excel_path = 'Questions_Jun_21.xlsx'
df = pd.read_excel(excel_path)


PATH="/Users/pavs/Dropbox/ChatGPT/Project two/paper Pavs/"
for folder in os.listdir(PATH):
    for filename in os.listdir(PATH+folder):
        if filename.endswith('.md'):
            # Construct full file path
            paper_path = os.path.join(PATH + folder + '/' + filename)
            pmid = filename.split(".")[0]
            pmid = pmid.split("(")[0]
            os.mkdir(PATH+folder+'/output/')
            save_path = PATH + folder + '/output/'+pmid+'_answers_Jun24.xlsx'
            print(paper_path,save_path)
            with open(paper_path, 'r', encoding='utf-8') as file:
                paper_content = file.read()
                query_gpt(system_prompt, paper_content, df, save_path)
