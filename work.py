import nest_asyncio
from dotenv import load_dotenv
from dotenv import find_dotenv
load_dotenv(find_dotenv())
import os
from llama_parse import LlamaParse
from pathlib import Path
import time

import pdf_utils 


nest_asyncio.apply()

WS = Path(__file__).resolve().parent
PAPER_PATH = WS / 'papers'


def get_papers():
    papers = []
    for i in PAPER_PATH.iterdir():      
        if not i.is_dir():
            continue

        for j in i.iterdir():
            if j.suffix != '.pdf':
                continue
            # if j.stem != i.stem:
            #     raise Exception('PMID of PDF is not the same as the folder.')
    
            papers.append(j)
    papers = sorted(papers)
    print(papers)
    return papers


def get_markdown_path(pdf_path):
    return pdf_path.parent / f"{pdf_path.name.replace('.pdf', '.markdown')}"


def save_markdown(markdown_path, markdown):
    with open(markdown_path, 'w') as fd:
        fd.write(markdown)


def convert_paper(parser_gpt4o, paper_path, gpt_mode):
    print(paper_path)
    # Get the parent folder name
    pmid = Path(paper_path).parent.name
    individual_pages_path = paper_path.parent / "individual_pages" 

    pdf_utils.split_pdf(pmid, paper_path, individual_pages_path)

    # for each page in pdf
    for page in individual_pages_path.iterdir():
        if page.suffix != '.pdf':
            continue
        print(page)
        start = time.time()
        documents = parser_gpt4o.load_data(str(page))
        stop = time.time()
        print(stop - start)

    # sync batch
    # documents = parser.load_data(["./my_file1.pdf", "./my_file2.pdf"])

    # async
    # documents = await parser.aload_data("./my_file.pdf")

    # async batch
    # documents = await parser.aload_data(["./my_file1.pdf", "./my_file2.pdf"])

        markdown = documents[0].get_content()
        markdown_path = get_markdown_path(page)
        save_markdown(markdown_path, markdown)
    
    pdf_utils.merge_markdown(pmid, gpt_mode)

def work():
    gpt_mode = False
    parser_gpt4o = LlamaParse(
        api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
        result_type="markdown",
        gpt4o_mode=gpt_mode,
        gpt4o_api_key=os.getenv('GPT-4O-KEY'),

        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

    paper_list = get_papers()
    print(paper_list)
    for p in paper_list:
        convert_paper(parser_gpt4o, p, gpt_mode)


if __name__ == '__main__':
    # create folder with same pmid for each pdf
    pdf_utils.organize_pdfs(PAPER_PATH)
    work()
