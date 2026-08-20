# Corrective RAG System

An end-to-end Corrective Retrieval-Augmented Generation pipeline: it answers
questions from your documents, grades retrieved chunks for relevance, and
automatically rewrites the query and re-retrieves when the first pass isn't
good enough — reducing hallucination by generating answers only from
verified context.

## How it works

```
question
   │
   ▼
retrieve (top-k from Chroma)
   │
   ▼
grade (LLM scores each chunk: relevant / irrelevant)
   │
   ├── enough relevant chunks? ──► generate answer (with sources)
   │
   └── not enough? ──► rewrite query ──► retrieve again (loop, up to N times)
                              │
                              └── still not enough? ──► optional web search ──► generate
```

If nothing verified is ever found, the system answers:
*"I don't have enough verified information to answer this question."*
instead of guessing.

## Project structure

```
corrective-rag/
├── app.py                 # Streamlit UI
├── requirements.txt
├── .env.example            # copy to .env and fill in your API key
├── data/                    # uploaded/ingested source documents land here
├── vectorstore/             # persisted Chroma DB (created automatically)
└── src/
    ├── config.py            # models, paths, thresholds — tune here
    ├── ingest.py             # load, clean, split, embed, store
    ├── grader.py              # relevance grading + query rewriting
    ├── generate.py             # grounded answer generation
    └── graph.py                # LangGraph flow tying it all together
```

## Setup

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your API key:**
   ```bash
   cp .env.example .env
   # then edit .env and set OPENAI_API_KEY=sk-...
   ```
   `TAVILY_API_KEY` is optional — only needed if you want the web-search
   fallback for when local documents have nothing relevant. Get a free key
   at https://tavily.com. Leave it blank to disable the fallback.

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Then open the local URL Streamlit prints (usually http://localhost:8501).

## Using it

1. In the sidebar, upload one or more PDF / DOCX / TXT files and click
   **Ingest documents**. This chunks and embeds them into a local Chroma
   vector store under `vectorstore/`.
2. Ask a question in the chat box. You'll see:
   - The answer, grounded only in verified chunks
   - Which source files it came from
   - Whether the query had to be rewritten and re-retrieved, and whether
     the web fallback was used

## Command-line usage (no UI)

```bash
# Ingest everything currently in data/
python -m src.ingest

# Ask a question from the terminal
python -m src.graph
```

## Tuning

Open `src/config.py`:

- `RELEVANCE_THRESHOLD` — fraction of retrieved chunks that must be graded
  relevant before the system trusts the retrieval (default 0.5).
- `MAX_REWRITE_ATTEMPTS` — how many times to rewrite + re-retrieve before
  giving up or falling back to web search (default 2).
- `TOP_K` — how many chunks to retrieve per pass (default 4).
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking granularity for ingestion.
- `LLM_MODEL` / `EMBEDDING_MODEL` — swap models here.

## Notes

- The vector store is persisted locally in `vectorstore/` — delete that
  folder to start fresh.
- Grading, rewriting, and generation all use the same small/cheap model
  (`gpt-4o-mini` by default) to keep cost down; swap to a stronger model in
  `config.py` if you need better grading accuracy.
