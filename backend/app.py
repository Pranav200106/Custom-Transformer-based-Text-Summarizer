from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import cross_origin, CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/summarize", methods=['POST'])
def summarize():
    text = request.json["text"]
    print(text)
    if not text.strip():
        return jsonify({"error": "No input provided"}), 400
    
    text = text[:1024]

    try:
        summary = summarizer(text, max_length=30, min_length = 20,do_sample=False)
        return jsonify({"summary": summary[0]["summary_text"]})
    except Exception as e:
        return jsonify({"error": "Failed to generate summary"}), 500


if __name__ == '__main__':
    app.run(debug=True)