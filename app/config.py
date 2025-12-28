import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "seu-ragbot"
PINECONE_ENV = "us-east-1"
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")
