import os
import sys
import json
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from rag_engine import PureRAGEngine
from rag_engine_ollama import PureRAGOllamaEmbeddingEngine

load_dotenv()

app = Flask(__name__)
engine = None


def build_engine():
    engine_mode = os.getenv('RAG_ENGINE', 'google').strip().lower()
    if engine_mode == 'ollama':
        return PureRAGOllamaEmbeddingEngine()
    if engine_mode == 'google':
        return PureRAGEngine()
    raise ValueError("Invalid RAG_ENGINE. Use 'google' or 'ollama'.")

def get_engine():
    global engine
    if engine is None:
        engine = build_engine()
    return engine

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    use_rag = data.get('use_rag', True) # Default to True
    history = data.get('history', [])
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        result = get_engine().generate(question, use_rag=use_rag, history=history)
        return jsonify({
            "answer": result['answer'],
            "sources": result['sources']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query/stream', methods=['POST'])
def query_stream():
    data = request.json
    question = data.get('question')
    mode = data.get('mode')
    # Backward compat: if mode not provided, derive from use_rag
    if mode is None:
        use_rag = data.get('use_rag', True)
        mode = "our_rag" if use_rag else "no_rag"
    history = data.get('history', [])

    if not question:
        return jsonify({"error": "No question provided"}), 400

    def event_stream():
        try:
            for event in get_engine().generate_stream(question, history=history, mode=mode):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
