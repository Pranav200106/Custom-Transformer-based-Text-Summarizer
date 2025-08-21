from flask import Flask, request, jsonify
from flask_cors import CORS
from model.inference import summarize

app = Flask(__name__)
CORS(app)

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        summary = summarize(data['text'])
        return jsonify({'summary': summary})
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({'error': 'Failed to generate summary'}), 500

if __name__ == '__main__':
    app.run(debug=True)
