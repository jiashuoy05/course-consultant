import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings

from rag_engine import BaseRAGEngine
from pipeline_utils import QueryExpander, DocumentReranker

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("API_KEY")

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text-v2-moe"


class PureRAGOllamaEmbeddingEngine(BaseRAGEngine):
    def __init__(self, index_path: str = None):
        if index_path is None:
            index_path = os.path.join(os.path.dirname(__file__), "../data/faiss_index/pure_rag_ollama")

        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        if not ollama_base_url:
            raise ValueError("Missing OLLAMA_BASE_URL. Set it in your .env file.")

        api_key = os.getenv("API_KEY")
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=ollama_base_url,
        )
        self.vectorstore = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0,
            max_output_tokens=2048,
            google_api_key=api_key,
        )

        # Initialize query expansion and reranking components
        self.query_expander = QueryExpander(api_key=api_key)
        self.document_reranker = DocumentReranker(api_key=api_key)


if __name__ == "__main__":
    engine = PureRAGOllamaEmbeddingEngine()
    print("Testing RAG mode (Ollama embeddings + Gemini generation)...")
    print(engine.generate("對 LLM 有興趣可以上什麼課？", use_rag=True)["answer"])
