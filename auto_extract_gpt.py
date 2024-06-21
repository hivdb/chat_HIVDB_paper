import google.generativeai as genai
from google.generativeai.types import safety_types
import os
from dotenv import load_dotenv
from dotenv import find_dotenv
load_dotenv(find_dotenv())
from pathlib import  Path
import pandas as pd
import time
from openai import OpenAI


WS = Path(__file__).parent
PAPERS = WS / 'database' / 'papers'


def query_gpt4o(system_prompt, prompt_text):

    client = OpenAI(api_key=os.getenv('GPT-4O-KEY'))
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]
    )

    resp = completion.choices[0].message.content

    return resp


def load_markdown():

    files = []

    for i in PAPERS.iterdir():
        if not i.is_dir():
            continue

        for j in i.iterdir():
            if 'checked' not in j.name:
                continue
            if j.suffix == '.markdown':
                files.append(j)
            elif j.suffix == '.md':
                files.append(j)

    return files


def extract_one_paper(
        content,
        system_prompt,
        user_prompt,
        questions,
        save_path):
    for i in range(len(questions["question"])):

        formatted_user_prompt = user_prompt.format(
            paper_content=content,
            question=questions["question"][i])

        resp = query_gpt4o(system_prompt, formatted_user_prompt)

        questions.at[i, 'Answer'] = resp

        time.sleep(2)

    questions.to_excel(save_path, index=False)


def extract_papers(files, system_prompt, user_prompt, questions):

    for f in files:
        print(f.name)
        save_path = f.parent / f'{f.parent.name}_related_sentence.xlsx'
        with open(f, encoding='utf-8') as fd:
            content = fd.read()

        if save_path.exists():
            continue

        extract_one_paper(
            content, system_prompt,
            user_prompt, questions, save_path)
        break


def work():

    with open(WS / 'prompt' / 'system.txt') as fd:
        system_prompt = fd.read()

    with open(WS / 'prompt' / 'auto_extract.txt') as fd:
        user_prompt = fd.read()

    questions = pd.read_excel(str(WS / 'database' / 'Questions.xlsx'))

    files = load_markdown()

    extract_papers(files, system_prompt, user_prompt, questions)


if __name__ == '__main__':
    work()
