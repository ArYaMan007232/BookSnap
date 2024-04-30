from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import textwrap

app = Flask(__name__)

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("pszemraj/led-large-book-summary")
model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/led-large-book-summary")
summarization_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer, return_text=True)

@app.route('/summarize', methods=['POST'])
def summarize_text():
    # Get the text from the request
    data = request.json
    text = data.get('text')

    # Generate the summary
    summary = summarization_pipeline(text)[0]['summary_text']
    wrapped_summary = textwrap.fill(summary, width=80)

    # Return the summary as a JSON response
    return jsonify({'summary': wrapped_summary})

if __name__ == "__main__":
    app.run(debug=True)
