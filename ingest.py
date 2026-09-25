from pathlib import Path
import hashlib
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOC_DIR = Path("data/documents")
USER_DOC_DIR = Path("data/user_documents")
DB_DIR = "chroma_db"
COLLECTION = "institutional_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"


def extract_pdf(path: Path):
    records = []
    reader = PdfReader(str(path))
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            records.append({"text": text, "source": path.name, "page": page_no})
    return records


def read_pdfs(paths):
    records = []
    for path in paths:
        records.extend(extract_pdf(Path(path)))
    return records


def index_paths(paths, reset=False):
    """Incrementally index PDFs. Existing chunks are preserved unless reset=True."""
    paths = [Path(p) for p in paths]
    raw = read_pdfs(paths)
    if not raw:
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks, metadatas, ids = [], [], []
    for rec in raw:
        for i, chunk in enumerate(splitter.split_text(rec["text"])):
            if len(chunk.strip()) < 40:
                continue
            chunks.append(chunk)
            metadatas.append({"source": rec["source"], "page": rec["page"], "chunk": i})
            ids.append(hashlib.sha1(
                f'{rec["source"]}:{rec["page"]}:{i}:{chunk}'.encode("utf-8")
            ).hexdigest())

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
    client = chromadb.PersistentClient(path=DB_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    # upsert makes repeated uploads safe and prevents duplicate-ID failures.
    col.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
    return len(raw), len(chunks)


def main():
    paths = sorted(DOC_DIR.glob("*.pdf"))
    if not paths:
        raise SystemExit("No PDFs found in data/documents/")
    pages, chunks = index_paths(paths, reset=True)
    print(f"Loaded {pages} pages -> {chunks} chunks")
    print(f"Indexed {chunks} chunks into {COLLECTION}")


if __name__ == "__main__":
    main()
