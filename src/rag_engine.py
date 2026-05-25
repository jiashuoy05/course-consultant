import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from pipeline_utils import QueryExpander, DocumentReranker

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("API_KEY")


class BaseRAGEngine:
    rag_prompt = ChatPromptTemplate.from_template("""
你是一個專業的校務諮詢助手。請根據以下【參考資料】回答使用者的【問題】。

【回覆準則】:
1. **事實準確性**: 關於校園規章、修課學分、行政程序、獎學金申請等事實，請嚴格遵守【參考資料】。
2. **優先級**: 如果問題涉及「獎學金」、「補助」、「急難救助」等行政事務，請優先參考【參考資料】中來源為 `.txt` 檔案或類型為 `administrative` 的內容。
3. **查無資訊時的處理**: 如果資料中確實找不到具體項目，請明確告知。但請注意，參考資料中可能包含多個片段，請仔細比對所有片段。
4. **通識與職涯建議**: 若涉及課程與工作的關係，運用你的 general knowledge 提供建議。
5. **出處**: 在回答結尾列出參考的資料來源。

【對話歷史】:
{chat_history}

【參考資料】:
{context}

【問題】:
{question}

你的回答:
""")

    no_rag = ChatPromptTemplate.from_template("""
你是一個校務諮詢助手。請直接根據你的內在知識回答問題。
注意：不要參考任何外部搜尋結果或檢索到的文件。

【對話歷史】:
{chat_history}

【問題】:
{question}

你的回答:
""")

    def retrieve(self, query: str, k: int = 5, use_expansion: bool = True, chat_history: str = "") -> List[Document]:
        """
        Enhanced retrieval with query expansion and reranking.

        Process:
        1. Expand query into 4 variants (original + 3 HyDE passages) using QueryExpander
        2. For each variant, retrieve top 10 chunks from vector store
        3. Rerank all collected documents against original query
        4. Return top-k documents after reranking

        Args:
            query: Original query string
            k: Number of final documents to return (default: 5)
            use_expansion: Whether to use query expansion (default: True)
            chat_history: Formatted conversation history for personalized HyDE expansion

        Returns:
            List of top-k documents sorted by relevance
        """
        if use_expansion:
            try:
                # Step 1: Expand query into 4 variants (original + 3 HyDE passages)
                queries = self.query_expander.expand_query(query, chat_history=chat_history)

                # Step 2: Retrieve top 10 for each query variant
                all_docs = []
                for q in queries:
                    docs = self.vectorstore.similarity_search(q, k=10)
                    all_docs.extend(docs)

                # Step 3: Rerank against original query
                reranked_docs = self.document_reranker.rerank(query, all_docs, top_k=k)

                return reranked_docs
            except Exception as e:
                # Fallback to simple retrieval if expansion/reranking fails
                print(f"Enhanced retrieval failed: {e}. Falling back to simple retrieval.")
                return self.vectorstore.similarity_search(query, k=k)
        else:
            # Simple similarity search (fallback)
            return self.vectorstore.similarity_search(query, k=k)

    def generate(self, query: str, use_rag: bool = True, history: List[Dict] = None) -> Dict[str, Any]:
        if use_rag:
            # RAG mode: include chat history, use enhanced retrieval
            chat_history = ""
            if history:
                for msg in history:
                    role = "使用者" if msg.get("role") == "user" else "AI"
                    chat_history += f"{role}: {msg.get('content')}\n"
            if not chat_history:
                chat_history = "無"

            docs = self.retrieve(query, k=5, use_expansion=True, chat_history=chat_history)
            context = "\n\n".join(
                [f"--- 來源: {doc.metadata.get('source')} ---\n{doc.page_content}" for doc in docs]
            )
            chain = self.rag_prompt | self.llm
            response = chain.invoke({"context": context, "question": query, "chat_history": chat_history})
            sources = list(set([doc.metadata.get("source") for doc in docs]))
        else:
            # No-RAG mode: one-shot, no history
            chain = self.no_rag | self.llm
            response = chain.invoke({"question": query, "chat_history": ""})
            sources = ["Internal Weights"]

        return {
            "answer": self._to_text(response.content),
            "sources": sources,
        }

    def generate_stream(self, query: str, use_rag: bool = True, history: List[Dict] = None, mode: str = None):
        """Generator yielding SSE event dicts for verbose pipeline display.

        mode: "no_rag" | "traditional_rag" | "our_rag"
        Falls back to use_rag if mode is not provided.
        """
        if mode is None:
            mode = "our_rag" if use_rag else "no_rag"

        if mode == "our_rag":
            chat_history = ""
            if history:
                for msg in history:
                    role = "使用者" if msg.get("role") == "user" else "AI"
                    chat_history += f"{role}: {msg.get('content')}\n"
            if not chat_history:
                chat_history = "無"

            yield {"step": "expand", "status": "running", "label": "重寫查詢", "data": None}
            try:
                queries = self.query_expander.expand_query(query, chat_history=chat_history)
            except Exception:
                queries = [query, query, query, query]
            yield {"step": "expand", "status": "done", "label": "重寫查詢",
                   "data": {"variants": [query] + queries}}

            yield {"step": "retrieve", "status": "running", "label": "向量檢索", "data": None}
            all_docs = []
            for q in [query] + queries:
                docs = self.vectorstore.similarity_search(q, k=10)
                all_docs.extend(docs)
            yield {"step": "retrieve", "status": "done", "label": "向量檢索",
                   "data": {"doc_count": len(all_docs)}}

            yield {"step": "rerank", "status": "running", "label": "重新排序", "data": None}
            try:
                reranked = self.document_reranker.rerank(query, all_docs, top_k=5)
            except Exception:
                reranked = all_docs[:5]
            yield {"step": "rerank", "status": "done", "label": "重新排序",
                   "data": {"top_k": len(reranked)}}

            yield {"step": "llm", "status": "running", "label": "LLM 思考中", "data": None}
            context = "\n\n".join(
                [f"--- 來源: {doc.metadata.get('source')} ---\n{doc.page_content}" for doc in reranked]
            )
            chain = self.rag_prompt | self.llm
            response = chain.invoke({"context": context, "question": query, "chat_history": chat_history})
            sources = list(set([doc.metadata.get("source") for doc in reranked]))
            yield {"type": "result", "answer": self._to_text(response.content), "sources": sources}

        elif mode == "traditional_rag":
            chat_history = ""
            if history:
                for msg in history:
                    role = "使用者" if msg.get("role") == "user" else "AI"
                    chat_history += f"{role}: {msg.get('content')}\n"
            if not chat_history:
                chat_history = "無"

            yield {"step": "retrieve", "status": "running", "label": "向量檢索", "data": None}
            docs = self.vectorstore.similarity_search(query, k=5)
            yield {"step": "retrieve", "status": "done", "label": "向量檢索",
                   "data": {"doc_count": len(docs)}}

            yield {"step": "llm", "status": "running", "label": "LLM 思考中", "data": None}
            context = "\n\n".join(
                [f"--- 來源: {doc.metadata.get('source')} ---\n{doc.page_content}" for doc in docs]
            )
            chain = self.rag_prompt | self.llm
            response = chain.invoke({"context": context, "question": query, "chat_history": chat_history})
            sources = list(set([doc.metadata.get("source") for doc in docs]))
            yield {"type": "result", "answer": self._to_text(response.content), "sources": sources}

        else:  # no_rag
            yield {"step": "llm", "status": "running", "label": "LLM 思考中", "data": None}
            chain = self.no_rag | self.llm
            response = chain.invoke({"question": query, "chat_history": ""})
            yield {"type": "result", "answer": self._to_text(response.content), "sources": ["Internal Weights"]}

    @staticmethod
    def _to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
            if parts:
                return "\n".join(parts)
        return str(content)


class PureRAGEngine(BaseRAGEngine):
    def __init__(self, index_path: str = None):
        if index_path is None:
            index_path = os.path.join(os.path.dirname(__file__), "../data/faiss_index/pure_rag")

        api_key = os.getenv("API_KEY")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=api_key,
        )
        self.vectorstore = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0,
            max_output_tokens=4096,
            google_api_key=api_key,
        )

        # Initialize query expansion and reranking components
        self.query_expander = QueryExpander(api_key=api_key)
        self.document_reranker = DocumentReranker(api_key=api_key)


if __name__ == "__main__":
    engine = PureRAGEngine()
    print("Testing RAG mode...")
    print(engine.generate("對 LLM 有興趣可以上什麼課？", use_rag=True)["answer"])
    print("\nTesting No-RAG mode...")
    print(engine.generate("對 LLM 有興趣可以上什麼課？", use_rag=False)["answer"])
