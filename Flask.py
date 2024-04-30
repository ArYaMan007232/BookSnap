from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import textwrap

app = Flask(__name__)

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("pszemraj/led-large-book-summary")
model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/led-large-book-summary")
summarization_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer, return_text=True)

def chunk_text(text, chunk_size=1024):
    # Chunk the text into smaller parts
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

@app.route('/summarize', methods=['POST'])
def summarize_text():
    # Get the text from the request
    data = request.json
    text = data.get('text')

    # Chunk the text into smaller parts
    chunks = chunk_text(text)

    # Generate summaries for each chunk
    summaries = []
    for chunk in chunks:
        summary = summarization_pipeline(chunk)[0]['summary_text']
        summaries.append(summary)

    # Combine the summaries into a single text
    final_summary = " ".join(summaries)
    wrapped_summary = textwrap.fill(final_summary, width=80)

    # Return the summary as a JSON response
    return jsonify({'summary': wrapped_summary})

if __name__ == "__main__":
    app.run(debug=True)

