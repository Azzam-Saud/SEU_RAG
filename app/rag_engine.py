from functools import lru_cache
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone

from app.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")

def search(query, k=15):
    model = get_model()

    q_emb = model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )[0]

    res = index.query(
        vector=q_emb.tolist(),
        top_k=k,
        include_metadata=True
    )

    results = []
    for match in res["matches"]:
        meta = match.get("metadata", {})
        results.append({
            "score": float(match["score"]),
            "text": meta.get("preview", ""),
            "source_id": meta.get("source_id")
        })

    return results

def rag_llm_answer(query: str):
    results = search(query)
    context = "\n\n".join(r["text"] for r in results)

    prompt = f"""
أجب فقط من النصوص التالية.
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
