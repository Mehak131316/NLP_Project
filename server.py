import os
import sys
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("Loading Contract-Sentry NLP modules and document processor...")
from document_processor import process_single_document, nlp, custom_tokenize, pipeline_final, tag_spacy_doc

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'NLP_Assessment1-20261003T034029Z-1-001', 'NLP_Assessment1', 'data')

@app.route('/api/search', methods=['GET'])
def search_query():
    q = request.args.get('q', '')
    pipeline = request.args.get('pipeline', 'Final Pipeline')
    try:
        # Fast query processor
        import backend
        ranked_docs, ms = backend.search(q, pipeline)
        q_type = backend.query_type(q)
        return jsonify({
            'query': q,
            'type': q_type,
            'pipeline': pipeline,
            'results': ranked_docs,
            'time_ms': round(ms, 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.json or {}
    text = data.get('text', '')
    try:
        t_start = time.perf_counter()
        
        # 1. Raw Tokenization timing
        t0 = time.perf_counter()
        raw = custom_tokenize(text)
        raw_list = [t[0] if isinstance(t, tuple) else t for t in raw]
        raw_time_ms = round((time.perf_counter() - t0) * 1000, 3)
        
        # 2. Final Pipeline timing (Lemmatization & Stop-words)
        t1 = time.perf_counter()
        final = pipeline_final(text) if callable(pipeline_final) else [t.lower() for t in raw_list]
        final_time_ms = round((time.perf_counter() - t1) * 1000, 3)
        
        # 3. POS Tagging timing
        t2 = time.perf_counter()
        pos_tags = tag_spacy_doc(final) if final else []
        pos_time_ms = round((time.perf_counter() - t2) * 1000, 3)
        
        total_time_ms = round((time.perf_counter() - t_start) * 1000, 3)
        
        return jsonify({
            'raw': raw_list,
            'raw_time_ms': raw_time_ms,
            'final': final,
            'final_time_ms': final_time_ms,
            'pos': pos_tags,
            'pos_time_ms': pos_time_ms,
            'total_time_ms': total_time_ms,
            'time_ms': total_time_ms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/available-docs', methods=['GET'])
def get_available_docs():
    """Return available contracts in the corpus data directory."""
    try:
        if os.path.exists(DATA_DIR):
            files = [
                {"name": f, "size_kb": round(os.path.getsize(os.path.join(DATA_DIR, f)) / 1024, 1)}
                for f in sorted(os.listdir(DATA_DIR))
                if f.lower().endswith(('.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg'))
            ]
            return jsonify({"files": files})
        return jsonify({"files": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-doc', methods=['POST'])
@app.route('/api/upload', methods=['POST'])
def process_document():
    """
    Process any PDF, DOCX, TXT or image through the full NLP pipeline
    and return complete statistics, stage-by-stage timings, entities, and n-grams.
    """
    try:
        # Check if file was uploaded via FormData
        if 'file' in request.files:
            uploaded_file = request.files['file']
            filename = uploaded_file.filename
            file_bytes = uploaded_file.read()
            result = process_single_document(file_bytes, filename)
            return jsonify(result)

        # Check if existing filename requested from corpus
        data = request.get_json(silent=True) or {}
        existing_filename = data.get('filename') or request.form.get('filename')
        
        if existing_filename:
            filepath = os.path.join(DATA_DIR, existing_filename)
            if not os.path.exists(filepath):
                return jsonify({"error": f"File not found: {existing_filename}"}), 404
            result = process_single_document(filepath, existing_filename)
            return jsonify(result)

        # Check if raw text was sent
        text_content = data.get('text')
        if text_content:
            doc_name = data.get('doc_name', 'Input_Document.txt')
            result = process_single_document(text_content.encode('utf-8'), doc_name)
            return jsonify(result)

        return jsonify({"error": "No file or text provided."}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("==================================================")
    print("Contract-Sentry NLP Backend is running on port 5000!")
    print("Full PDF and Multi-Format Document Processing Enabled")
    print("==================================================")
    app.run(port=5000, debug=False)
