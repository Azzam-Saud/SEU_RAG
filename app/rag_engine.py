import faiss
import pickle
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI()

INDEX_PATH = "index.faiss"
META_PATH = "meta.pkl"

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search(query, k=5):
    model = get_model()
    q_emb = model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )

    index, metadata = load_index()
    scores, ids = index.search(q_emb, k)

    results = []
    for i in ids[0]:
        if i == -1:
            continue
        results.append(metadata[i])

    return results

def rag_llm_answer(query: str):
    chunks = search(query)
    context = "\n\n".join(chunks)

    prompt = f"""
أجب فقط من النص التالي.
إذا لم تجد الإجابة قل: لا أعلم.

السؤال:
{query}

السياق:
{context}

الإجابة:
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content.strip()
