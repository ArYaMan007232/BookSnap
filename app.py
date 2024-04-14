import os
import textwrap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import codecs
import fitz  

tokenizer = AutoTokenizer.from_pretrained("pszemraj/led-large-book-summary")
model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/led-large-book-summary")

summarization_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer, return_text=True)

def extract_text_from_txt(txt_file):
    with open(txt_file, 'r', encoding="utf-8") as file:
        text = file.read()
    return text

def convert_pdf_to_text(pdf_file):
    text = ""
    with fitz.open(pdf_file) as pdf:
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

def chunk_text(text, max_chunk_len=400):
    chunks = []
    tokens = text.split()
    for i in range(0, len(tokens), max_chunk_len):
        chunks.append(' '.join(tokens[i:i+max_chunk_len]))
    return chunks

if _name_ == "_main_":
    input_file = input("Enter the path to the input file: ")

    if not os.path.exists(input_file):
        print("File not found.")
        exit()

    file_extension = os.path.splitext(input_file)[1].lower()

    if file_extension == ".pdf":
        txt_file = os.path.splitext(input_file)[0] + ".txt"

        pdf_text = convert_pdf_to_text(input_file)

        with open(txt_file, "w", encoding="utf-8") as txt_file_obj:
            txt_file_obj.write(pdf_text)

        text = extract_text_from_txt(txt_file)
    elif file_extension == ".txt":
        text = extract_text_from_txt(input_file)
    else:
        print("Unsupported file format. Please provide either a PDF or a text file.")
        exit()

    text_chunks = chunk_text(text)

    for chunk_num, chunk in enumerate(text_chunks):
        summary = summarization_pipeline(chunk)[0]['summary_text']
        wrapped_summary = textwrap.fill(summary, width=80)
        paragraphs = wrapped_summary.split('\n\n')
        for paragraph_num, paragraph in enumerate(paragraphs):
            wrapped_paragraph = textwrap.fill(paragraph, width=80)
            formatted_summary = f"Chunk {chunk_num+1}, Paragraph {paragraph_num+1}:\n{wrapped_paragraph}\n\n"
            print(formatted_summary)