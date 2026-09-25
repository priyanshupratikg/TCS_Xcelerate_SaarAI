# Institutional Knowledge Retrieval System — LLMOps Capstone

A fast MVP for TCS Industry-Aligned Capstone Use Case A.

## What this demonstrates
- PDF ingestion and chunking
- Sentence-transformer embeddings
- Persistent Chroma vector store
- Retrieval-Augmented Generation (RAG)
- Prompt versioning under Git
- Prompt-injection and out-of-scope guardrails
- Basic token/cost logging
- Golden-set evaluation with an optional LLM judge
- Streamlit dashboard

## 1. Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your Gemini API key in `.env`.

Google's current Python SDK is `google-genai`. The code uses the current `client.models.generate_content(...)` interface.

## 2. Add institutional PDFs
Replace the demo PDFs in `data/documents/` with at least five official institutional documents.

## 3. Build the vector database
```bash
python ingest.py
```

## 4. Run the app
```bash
streamlit run app.py
```

## 5. Run evaluation
```bash
python evaluate.py
```

## Demo questions
- What is the minimum attendance requirement?
- When are the semester examinations scheduled?
- What is the tuition fee?
- What documents are required for registration?
- What should I do if I have a fee-related query?

## Architecture
PDFs -> PyPDF -> chunking -> Sentence Transformers -> Chroma
                                              |
Question -> guardrails -> embedding -> top-k retrieval -> Gemini -> cited answer
                                              |
                                        logs/evaluation

## User PDF Upload & Multi-Document Knowledge Base

The Streamlit application supports a persistent user knowledge base:

1. Upload one or multiple PDFs from the sidebar.
2. Click **Add PDFs to Knowledge Base**.
3. PDFs are stored in `data/user_documents/` and indexed incrementally into ChromaDB.
4. Ask questions across the complete combined document collection.
5. Upload additional PDFs later without rebuilding the existing base.

This demonstrates a practical multi-document RAG workflow while keeping the source document name and page number in retrieval metadata.
