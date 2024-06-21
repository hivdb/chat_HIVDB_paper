import google.generativeai as genai
from google.generativeai.types import safety_types
import os
from dotenv import load_dotenv
from dotenv import find_dotenv
load_dotenv(find_dotenv())
from pathlib import  Path
import pandas as pd
import time


WS = Path(__file__).parent
PAPERS = WS / 'database' / 'papers'


def query_gemini(system_prompt, prompt_text):

    genai.configure(api_key=os.environ["GEMINI_KEY"])
    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        system_instruction=system_prompt,
        generation_config={
            'temperature': 0,
            'top_p': 1,
            'top_k': 1,
        },
        safety_settings=[
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
        ])

    response = model.generate_content(
        prompt_text,
        safety_settings=[
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
        ])  # not sure why a research study can trigger safety issue, thank you google.

    # RECITATION，what does it even mean? cannot find answer in document.
    # print(response)
    try:
        return response.text
    except ValueError:
        return ''


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


def query_one_paper(
        content,
        system_prompt,
        user_prompt,
        questions,
        save_path):
    for i in range(len(questions["question"])):

        formatted_user_prompt = user_prompt.format(
            paper_content=content,
            question=questions["question"][i],
            requirements=questions["Prompt_additional"][i])

        resp = query_gemini(system_prompt, formatted_user_prompt)

        answers = resp.split('Reference:', 1)

        questions.at[i, 'Answer'] = answers[0]
        questions.at[i, 'Reference Sentences'] = answers[1] if len(answers) > 1 else ''

        time.sleep(2)

    questions.to_excel(save_path, index=False)


def query_papers(files, system_prompt, user_prompt, questions):

    for f in files:
        print(f.name)
        save_path = f.parent / f'{f.parent.name}_answers_Gemini.xlsx'
        with open(f, encoding='utf-8') as fd:
            content = fd.read()

        if save_path.exists():
            continue

        query_one_paper(
            content, system_prompt,
            user_prompt, questions, save_path)


def work():

    with open(WS / 'prompt' / 'system.txt') as fd:
        system_prompt = fd.read()

    with open(WS / 'prompt' / 'user.txt') as fd:
        user_prompt = fd.read()

    question = pd.read_excel(str(WS / 'database' / 'Questions.xlsx'))

    files = load_markdown()

    query_papers(files, system_prompt, user_prompt, question)

    # answer = query_gemini(model, prompt_text)


if __name__ == '__main__':
    work()
