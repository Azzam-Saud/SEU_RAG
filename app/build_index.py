import hashlib
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from app.loaders import extract_txt_from_files, extract_word, extract_excel
from app.chunking import chunk_policy_qna_articles
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def safe_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

BASE_DIR = r"D:\Azzam\Personal_Projects\SEU\filtered_data"
DIR1 = os.path.join(BASE_DIR, "Word_Excel")
DIR2 = os.path.join(BASE_DIR, "txt")

records = []

def process_dir(folder):
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        ext = fname.split(".")[-1].lower()

        if ext == "txt":
            recs = extract_txt_from_files(path, fname)
        elif ext == "docx":
            recs = extract_word(path, fname)
        elif ext in ["xlsx", "xls"]:
            recs = extract_excel(path, fname)
        else:
            continue

        records.extend(recs)

process_dir(DIR1)
process_dir(DIR2)

# 🔹 Chunking
chunked = []
for rec in records:
    chunks = [rec["text"]] if rec["type"] == "excel" else chunk_policy_qna_articles(rec["text"])
    for i, ch in enumerate(chunks):
        chunked.append({
            "id": f'{rec["id"]}__chunk_{i}',
            "text": ch
        })

texts = [c["text"] for c in chunked]
ids = [c["id"] for c in chunked]

# 🔹 OpenAI Embeddings (1024 dims)
def embed_texts(texts):
    embeddings = []
    for t in texts:
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=t
        )
        embeddings.append(res.data[0].embedding)
    return embeddings

embs = embed_texts(texts)

# 🔹 Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(PINECONE_INDEX_NAME)

# 🔹 Upsert
MAX_META_CHARS = 2000

vectors = [
    {
        "id": safe_id(ids[i]),
        "values": embs[i],
        "metadata": {
            "preview": texts[i][:MAX_META_CHARS],
            "source_id": ids[i]
        }
    }
    for i in range(len(ids))
]

index.upsert(vectors=vectors, batch_size=100)

print("✅ Data uploaded to Pinecone successfully")
