import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from app.config import OPENAI_API_KEY
from functools import lru_cache

@lru_cache(maxsize=1)
def get_resources():
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )
    index = faiss.read_index("faiss.index")
    meta = pickle.load(open("meta.pkl", "rb"))
    return model, index, meta
    
meta = pickle.load(open("meta.pkl", "rb"))
texts, ids = meta["texts"], meta["ids"]

client = OpenAI(api_key=OPENAI_API_KEY)

def search(query, k=10):
    model, index, meta = get_resources()
    texts = meta["texts"]

    q_emb = model.encode(
        ["query: " + query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(q_emb, k)
    return [{"score": float(D[0][i]), "text": texts[idx]} for i, idx in enumerate(I[0])]
    
def rag_llm_answer(query: str):
    results = search(query)
    context = "\n\n".join(r["text"] for r in results)

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
