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


def rag_llm_answer(query: str):
    prompt = f"""
أجب على السؤال التالي مباشرة:

{query}
"""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    return extract_text(response)
