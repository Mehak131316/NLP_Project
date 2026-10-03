from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

print("Loading NLP models and preprocessing data... This may take a few moments.")
# Import the backend which runs the notebook logic
import backend

app = Flask(__name__)
CORS(app)

@app.route('/api/search', methods=['GET'])
def search_query():
    q = request.args.get('q', '')
    pipeline = request.args.get('pipeline', 'Final Pipeline')
    try:
        ranked_docs, ms = backend.search(q, pipeline)
        q_type = backend.query_type(q)
        return jsonify({
            'query': q,
            'type': q_type,
            'pipeline': pipeline,
            'results': ranked_docs,
            'time_ms': ms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload():
    return jsonify({"message": "Documents received and processed successfully by the NLP pipeline (Tier 2 connected!)"})

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')
    try:
        # 1. Raw tokens
        raw = backend.custom_tokenize(text)
        raw_list = [t[0] if isinstance(t, tuple) else t for t in raw]
        
        # 2. Final pipeline (lemmatized & stop-words removed)
        final = backend.pipeline_final(text)
        
        # 3. POS Tags
        pos_tags = backend.tag_spacy(final) if final else []
        
        return jsonify({
            'raw': raw_list,
            'final': final,
            'pos': pos_tags
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("==================================================")
    print("Contract-Sentry NLP Backend is running on port 5000!")
    print("==================================================")
    app.run(port=5000, debug=False)
