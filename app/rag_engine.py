from openai import OpenAI
from pinecone import Pinecone
from app.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def embed_query(text: str):
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return emb.data[0].embedding

def search(query, k=15):
    q_emb = embed_query(query)

    res = index.query(
        vector=q_emb,
        top_k=k,
        include_metadata=True
    )

    return [
        {
            "score": match["score"],
            "text": match["metadata"].get("preview", "")
        }
        for match in res["matches"]
    ]

def rag_llm_answer(query: str):
    results = search(query)
    context = "\n\n".join(r["text"] for r in results)

    prompt = f"""
أجب فقط من المستندات المتاحة
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
