import faiss
import pickle
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from app.config import OPENAI_API_KEY

@lru_cache(maxsize=1)
def get_resources():
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )
    index = faiss.read_index("faiss.index")
    meta = pickle.load(open("meta.pkl", "rb"))
    return model, index, meta


client = OpenAI(api_key=OPENAI_API_KEY)


def rag_llm_answer(query: str) -> str:
    model, index, meta = get_resources()
    texts = meta["texts"]

    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(q_emb, 5)
    context = "\n\n".join(texts[idx] for idx in I[0])

    prompt = f"""
أجب فقط من النص التالي.
إذا لم تجد الإجابة قل: لا أعلم.

السؤال:
{query}

السياق:
{context}

الإجابة:
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
        )

    # طريقة آمنة لاستخراج النص
    output_text = ""
    
    for item in response.output:
        if item["type"] == "message":
            for c in item["content"]:
                if c["type"] == "output_text":
                    output_text += c["text"]
    
    if not output_text.strip():
        return "لا أعلم."
    
    return output_text.strip()
