
import os
import glob
from pypdf import PdfReader, PdfWriter
import shutil
from pathlib import Path
import re

def split_pdf(filename, input_pdf_path, output_dir):
    # Create a PdfReader object
    reader = PdfReader(input_pdf_path)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all pages
    for i in range(len(reader.pages)):
        writer = PdfWriter()
        
        # Add a single page to the writer object
        writer.add_page(reader.pages[i])
        
        # Create a new output file name
        output_pdf_path = os.path.join(output_dir, f'{filename}_{i+1}.pdf')
        
        # Write the single page to a new PDF file
        with open(output_pdf_path, 'wb') as output_pdf_file:
            writer.write(output_pdf_file)
    
    print(f'Successfully split the PDF into {len(reader.pages)} pages.')

def organize_pdfs(pdf_directory):
    # Ensure the pdf_directory is a Path object
    pdf_directory = Path(pdf_directory)

    # Iterate over all PDF files in the directory
    for pdf_path in pdf_directory.glob("*.pdf"):
        # Get the stem (name without extension) of the PDF
        folder_name = pdf_path.stem
        
        # Create a new folder with the same name as the PDF file (excluding the .pdf extension)
        new_folder_path = pdf_directory / folder_name
        new_folder_path.mkdir(exist_ok=True)
        
        # Move the PDF file into the new folder
        new_pdf_path = new_folder_path / pdf_path.name
        shutil.move(str(pdf_path), str(new_pdf_path))
        
        print(f"Moved {pdf_path.name} to {new_folder_path}")

def merge_markdown(filename, gpt_mode):
    file_list = glob.glob(f'papers/{filename}/individual_pages/{filename}_*.markdown')
    mode = "No GPT-4o"
    content = ""
    if gpt_mode: mode = "GPT-4o"
    with open(f'papers/{filename}/{filename}({mode}).md', 'wb') as outfile:
        for filenum in sorted(file_list, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])):
            # print(filename)
            with open(filenum, 'rb') as infile:
                outfile.write(infile.read())
                outfile.write(b'\n') 
    # remove references
    pattern = re.compile(r'reference', re.IGNORECASE)
    with open(f'papers/{filename}/{filename}({mode}).md', 'r') as file:
        content = file.read()
        

    # Find all matches and print their positions
    for match in pattern.finditer(content):
        last_match = match

    last_match_position = last_match.start()

    if last_match_position != -1:
        # Truncate the content after the last occurrence of the search string
        new_content = content[:last_match_position]

    # Write the truncated content back to the file
    with open(f'papers/{filename}/{filename}({mode}).md', 'w') as file:
        file.write(new_content)