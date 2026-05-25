import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders import load_admin_data, load_preprocessed_courses

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("API_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ADMIN_DATA_DIR = os.path.join(DATA_DIR, "行政資料")
INDEX_SAVE_PATH = os.path.join(DATA_DIR, "faiss_index/pure_rag")


def main():
    admin_docs = load_admin_data(ADMIN_DATA_DIR)
    course_docs = load_preprocessed_courses(PROCESSED_DIR)

    docs = admin_docs + course_docs
    print(f"Total documents: {len(docs)} (Admin: {len(admin_docs)}, Course: {len(course_docs)})")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    split_docs = text_splitter.split_documents(docs)
    print(f"Total chunks: {len(split_docs)}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        google_api_key=os.getenv("API_KEY"),
    )

    vectorstore = None
    v_lock = threading.Lock()
    batch_size = 50
    batches = [split_docs[p : p + batch_size] for p in range(0, len(split_docs), batch_size)]

    def process_batch(idx_batch):
        idx, batch = idx_batch
        nonlocal vectorstore
        for attempt in range(3):
            try:
                tmp_vs = FAISS.from_documents(batch, embeddings)
                with v_lock:
                    if vectorstore is None:
                        vectorstore = tmp_vs
                    else:
                        vectorstore.merge_from(tmp_vs)
                if (idx + 1) % 10 == 0:
                    print(f"Batch {idx + 1}/{len(batches)} done.")
                return
            except Exception as e:
                if "429" in str(e):
                    time.sleep(10 * (attempt + 1))
                else:
                    print(f"Batch {idx + 1} error: {e}")
                    break

    print(f"Processing {len(batches)} batches with 3 workers...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_batch, enumerate(batches))

    if vectorstore:
        print(f"Saving to {INDEX_SAVE_PATH}...")
        os.makedirs(os.path.dirname(INDEX_SAVE_PATH), exist_ok=True)
        vectorstore.save_local(INDEX_SAVE_PATH)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
