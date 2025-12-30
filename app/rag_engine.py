import faiss
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from openai import OpenAI

INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.pkl"

client = OpenAI(api_key=OPENAI_API_KEY)

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")
def embed(texts):
    model = get_model()
    return model.encode(
        ["passage: " + t for t in texts],
        normalize_embeddings=True
    )
@lru_cache(maxsize=1)
def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def search(query, k=10):
    model = get_model()
    index, meta = load_index()

    q_emb = model.encode(
        ["query: " + query],
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q_emb, k)

    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "text": meta[idx]["text"],
            "source_id": meta[idx]["id"]
        })

    return results

def rag_llm_answer(query: str):
    docs = search(query)

    context = "\n\n".join(d["text"] for d in docs)

    prompt = f"""
أجب فقط من السياق التالي.
إذا لم تجد الإجابة قل: لا أعلم.

السؤال:
{query}

السياق:
{context}

الإجابة:
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return res.choices[0].message.content.strip()



