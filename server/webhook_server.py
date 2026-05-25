import sys
import os
from flask import Flask, request, jsonify

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import PureRAGEngine

app = Flask(__name__)

# Global engine instance for the webhook server
# Using a global instance to avoid reloading the FAISS index on every request
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        print("正在初始化 Pure RAG 引擎...")
        _engine = PureRAGEngine()
    return _engine

# Check configuration on startup
try:
    print("正在準備 Webhook 服務...")
    get_engine()
    print("服務準備就緒！")
except Exception as e:
    print(f"初始化失敗: {e}")

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if not data:
        return "Invalid JSON", 400
        
    print(f"Received webhook data: {data}")
    
    # 按照需求：'text' 為輸入，'title' 是發送者（忽略）
    user_input = data.get('text')
    
    if user_input:
        user_input = user_input.strip()
    
    if not user_input:
        return "Empty message (expected 'text' field)", 400
        
    try:
        print(f"Processing request: {user_input}")
        # 使用 Pure RAG 引擎生成回答 (預設開啟 RAG)
        result = get_engine().generate(user_input, use_rag=True)
        
        # Webhook 通常預期直接返回文字訊息
        return result['answer']
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    # 預設在 5000 埠運行
    app.run(host='0.0.0.0', port=5000)
