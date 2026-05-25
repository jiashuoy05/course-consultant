"""
Pipeline utilities for enhanced RAG retrieval.

Includes query expansion and document reranking.
"""

import re
import json
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


class QueryExpander:
    """
    Expand a query into multiple paraphrased variants personalized with
    student context extracted from the conversation history.
    """

    @staticmethod
    def _content_to_str(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if text:
                    parts.append(str(text))
            return "\n".join(parts)
        return str(content)

    def __init__(self, api_key: str = None, model: str = "gemini-3.1-flash-lite-preview"):
        """
        Initialize QueryExpander with Gemini API.

        Args:
            api_key: Google API key for Gemini
            model: Model name to use (default: gemini-3.1-flash-lite-preview)
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.7,
            max_output_tokens=600,
            google_api_key=api_key,
        )

        # Prompt: extract student context, then produce 3 paraphrased question variants
        self.prompt = PromptTemplate(
            input_variables=["query", "chat_history"],
            template="""You are a retrieval assistant for a university course consultation system.

From the conversation history, silently note any student personal details (major, year, interests, GPA, etc.) and use them to make the paraphrases more specific and relevant to that student.

Your ONLY output must be exactly 3 alternative phrasings of the student's query. Each rephrasing should:
- Be a natural question or search phrase (NOT a document excerpt or answer)
- Incorporate relevant student details from history when helpful
- Approach the same information need from a slightly different angle

Output only the 3 rephrasings, one per line, nothing else.

Conversation History:
{chat_history}

Student Query: {query}

Output (3 rephrasings, one per line):""",
        )

    def expand_query(self, query: str, chat_history: str = "") -> List[str]:
        """
        Expand a query into 4 variants: original + 3 paraphrased versions.

        Args:
            query: Original query string
            chat_history: Formatted conversation history string

        Returns:
            List of 4 items: [original_query, rephrasing_1, rephrasing_2, rephrasing_3]
        """
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"query": query, "chat_history": chat_history or "無"})
            response_text = self._content_to_str(response.content).strip()

            # Strip numbered/bulleted prefixes like "1.", "2.", "-", "*"
            cleaned = []
            for line in response_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r'^[\d]+[.)]\s*', '', line)
                line = re.sub(r'^[-*•]\s*', '', line).strip()
                if line:
                    cleaned.append(line)

            passages = cleaned[:3]
            if len(passages) < 1:
                return [query, query, query, query]
            # Pad with original if fewer than 3 paraphrases were returned
            while len(passages) < 3:
                passages.append(query)
            return [query] + passages

        except Exception as e:
            print(f"Query expansion failed: {e}. Using original query.")
            return [query, query, query, query]


class DocumentReranker:
    """Rerank documents using Gemini LLM for listwise semantic relevance scoring."""

    def __init__(self, api_key: str = None, model: str = "gemini-3.1-flash-lite-preview"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            max_output_tokens=200,
            google_api_key=api_key,
        )
        self.prompt = PromptTemplate(
            input_variables=["query", "documents", "top_k"],
            template="""You are a document relevance ranker for a university course consultation system.

Given the student query and numbered document chunks below, return the indices of the {top_k} most relevant chunks, ordered from most to least relevant.

Student Query: {query}

Documents:
{documents}

Output only a JSON array of {top_k} integer indices (most relevant first). Example: [2, 0, 5, 1, 3]
Output:""",
        )

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return []

        # Deduplicate by page_content (4 query variants × 10 docs = many duplicates)
        seen: set = set()
        unique_docs: List[Document] = []
        for doc in documents:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        candidates = unique_docs[:20]  # Cap at 20 to keep prompt size reasonable

        if len(candidates) <= top_k:
            return candidates

        doc_text = "\n\n".join(
            f"[{i}] {doc.page_content[:400]}" for i, doc in enumerate(candidates)
        )

        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"query": query, "documents": doc_text, "top_k": top_k})
            raw = response.content if isinstance(response.content, str) else (
                "\n".join(
                    (item.get("text") if isinstance(item, dict) else getattr(item, "text", "")) or ""
                    for item in response.content
                ) if isinstance(response.content, list) else str(response.content)
            )
            match = re.search(r'\[[\d,\s]+\]', raw.strip())
            if match:
                indices = json.loads(match.group())
                ranked = [candidates[i] for i in indices if 0 <= i < len(candidates)]
                if ranked:
                    return ranked[:top_k]
        except Exception as e:
            print(f"LLM reranking failed: {e}. Falling back to original order.")

        return candidates[:top_k]
