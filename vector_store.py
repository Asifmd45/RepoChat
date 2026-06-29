import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vector_store(chunks: list[Document], repo_name: str) -> Chroma:
    embeddings = get_embeddings()
    collection_name = repo_name.replace("/", "_").replace("-", "_").lower()

    # wipe existing collection for this repo so re-ingestion is clean
    persist_dir = os.path.join(CHROMA_DIR, collection_name)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store built: {collection_name} — {len(chunks)} chunks indexed")
    return vector_store


def load_vector_store(repo_name: str) -> Chroma:
    embeddings = get_embeddings()
    collection_name = repo_name.replace("/", "_").replace("-", "_").lower()
    persist_dir = os.path.join(CHROMA_DIR, collection_name)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vector_store


def get_retriever(vector_store: Chroma, doc_type: str = None):
    if doc_type:
        # filtered retriever — only issues, only code, etc.
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 20,
                "lambda_mult": 0.7,
                "filter": {"type": doc_type}
            }
        )
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

# Quick test
if __name__ == "__main__":
    from github_ingestor import ingest_repo
    from chunker import chunk_documents

    raw_docs = ingest_repo("https://github.com/Asifmd45/Advanced-RAG-Pipeline")
    chunks = chunk_documents(raw_docs)

    vs = build_vector_store(chunks, "Asifmd45/Advanced-RAG-Pipeline")
    retriever = get_retriever(vs)

    results = retriever.invoke("how does the retrieval pipeline work")
    print(f"\nTop {len(results)} results for query: 'how does the retrieval pipeline work'")
    for r in results:
        print(f"\n  Source: {r.metadata['source']}")
        print(f"  Type: {r.metadata['type']}")
        print(f"  Preview: {r.page_content[:200]}")