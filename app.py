import os
import textwrap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import codecs
import fitz
from flask import Flask, request, jsonify

app = Flask(__name__)

# Disable TensorRT
os.environ['TF_TENSORRT_USE_FP16'] = '0'

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

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text')
    text_chunks = chunk_text(text)
    summaries = []
    for chunk_num, chunk in enumerate(text_chunks):
        summary = summarization_pipeline(chunk)[0]['summary_text']
        wrapped_summary = textwrap.fill(summary, width=80)
        summaries.append(wrapped_summary)
    return jsonify({'summaries': summaries})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
