import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Models ---
LLM_MODEL = "gpt-4o-mini"          # used for grading, rewriting, generation
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Paths ---
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "corrective_rag_docs"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval ---
TOP_K = 4

# --- Correction thresholds ---
# minimum fraction of retrieved chunks that must be graded "relevant"
# before we trust the retrieval and skip the rewrite/re-retrieve loop
RELEVANCE_THRESHOLD = 0.5
MAX_REWRITE_ATTEMPTS = 2

# Enable web search fallback if local docs still don't have the answer
# after the max rewrite attempts (requires TAVILY_API_KEY)
ENABLE_WEB_FALLBACK = bool(TAVILY_API_KEY)
