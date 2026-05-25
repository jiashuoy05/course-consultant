# 智慧創新大賞 - 個人 AI 選課諮詢顧問

> 2026 智慧創新大賞參賽作品

一款基於傳統 **RAG** 技術改良的校園選課諮詢系統，整合課程查詢、修課規範、行政法規與獎學金等校務資訊，讓使用者透過自然語言對話，快速獲得正確且個人化的課程建議。

## 功能特色

- **對話式問答**：支援多輪對話，系統自動從對話歷史推論使用者背景與需求，提供更精準的回答
- **可靠的資訊來源**：所有答案皆基於官方課程大綱、畢業規定與行政公告，不憑空捏造
- **個人化選課建議**：理解使用者需求後，主動推薦合適課程並說明理由
- **即時 Pipeline 視覺化**：介面即時呈現查詢擴展、向量檢索、重排序等各階段進度

## 系統架構

本系統實作三階段 Advanced RAG Pipeline：

```
使用者提問
    │
    ▼
┌─────────────────────────────┐
│  Pre-Retrieval              │
│  Query Expansion & Rewrite  │  → 生成 3 組改寫查詢（結合對話歷史）
└─────────────────────────────┘
    │ 4 組查詢（原始 + 3 改寫）
    ▼
┌─────────────────────────────┐
│  Retrieval                  │
│  FAISS 向量相似度檢索        │  → 每組查詢取 Top-10，共 40 筆候選
└─────────────────────────────┘
    │ 去重後最多 20 筆候選文件
    ▼
┌─────────────────────────────┐
│  Post-Retrieval             │
│  LLM-based Reranking        │  → HF/bge-reranker-v2-m3 列表式重排序，選出 Top-5
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Generation                 │
│  Gemini 答案生成             │  → 依據 Top-5 文件生成引用式回答
└─────────────────────────────┘
```

## 技術棧

| 層級 | 技術 |
|---|---|
| LLM & Embedding | Google Gemini API |
| RAG 框架 | LangChain |
| 向量資料庫 | FAISS |
| 後端 | Python 3.11 / Flask |
| 前端 | HTML / CSS / JavaScript（SSE 串流）|
| 本地 Embedding（替代）| Ollama + nomic-embed-text-v2-moe |

## 知識庫

| 資料類型 | 內容 |
|---|---|
| 課程大綱 | 大學部 / 研究所課程基本資訊與教學目標 |
| 畢業規定 | 各系所學分規範 |
| 行政公告 | 各行政單位公告（含 70+ 筆獎學金資訊）|

---

## 快速啟動

1. 建立並啟用虛擬環境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. 安裝依賴

```bash
pip install -r requirements.txt
```

3. 設定環境變數

```bash
cp .env.example .env
```

在 `.env` 中設定：

```env
API_KEY=your_google_ai_api_key
OLLAMA_BASE_URL=http://localhost:11434
RAG_ENGINE=google
```

4. 建立向量索引（首次或資料更新後）

```bash
python src/rag_ingestion.py
```

5. 啟動 Web UI

```bash
python rag_demo_app.py
```

6. （選用）啟動 Webhook 服務

```bash
python server/webhook_server.py
```

## Ollama 本地 Embedding 建置（替代方案）

如果你想改用本地 embedding（不使用 Google embedding）建立 FAISS 索引：

1. 啟動 Ollama 服務並拉取 embedding 模型

```bash
ollama pull nomic-embed-text-v2-moe
```

2. 在 `.env` 中設定 Ollama 服務位置（`OLLAMA_BASE_URL`）

3. 執行替代 ingestion

```bash
python src/rag_ingestion_ollama.py
```

預設會輸出到 `data/faiss_index/pure_rag_ollama`。
此腳本的 embedding 模型固定為 `nomic-embed-text-v2-moe`。

## 引擎切換（Google Embedding / Ollama Embedding）

Web UI 會依照 `.env` 的 `RAG_ENGINE` 載入不同引擎：

- `RAG_ENGINE=google`: 使用 `src/rag_engine.py`（Google embedding + Gemini）
- `RAG_ENGINE=ollama`: 使用 `src/rag_engine_ollama.py`（Ollama embedding + Gemini）

切到 Ollama embedding 時，請確認：

1. `OLLAMA_BASE_URL` 已正確設定
2. 已先執行 `python src/rag_ingestion_ollama.py` 建立索引

## 目前分支的服務模式

- Web UI: Flask + Jinja 模板（`rag_demo_app.py` + `templates/index.html`）
- Webhook: 獨立 Flask 路由（`server/webhook_server.py`）
- RAG 引擎: `src/rag_engine.py`
- 索引建置: `src/rag_ingestion.py`
