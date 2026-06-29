from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CODE_EXTENSIONS = [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", 
                   ".c", ".h", ".cs", ".go", ".rs", ".rb", ".php", ".swift", 
                   ".kt", ".sh"]

def get_splitter(doc_type: str, extension: str = ""):
    if doc_type == "code" or extension in CODE_EXTENSIONS:
        # smaller chunks for code — functions/classes fit in ~400 tokens
        return RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
        )
    elif doc_type == "markdown":
        # markdown splits cleanly by headers and paragraphs
        return RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )
    else:
        # issues and PRs are short — one or two chunks max
        return RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )


def chunk_documents(raw_docs: list[dict]) -> list[Document]:
    all_chunks = []

    for doc in raw_docs:
        content = doc["content"]
        metadata = doc["metadata"]

        if not content.strip():
            continue

        splitter = get_splitter(
            doc_type=metadata.get("type", ""),
            extension=metadata.get("extension", "")
        )

        chunks = splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            all_chunks.append(Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_index": i
                }
            ))

    return all_chunks


# Quick test
if __name__ == "__main__":
    from github_ingestor import ingest_repo

    raw_docs = ingest_repo("https://github.com/Sushma-1706/RepoRefine")
    chunks = chunk_documents(raw_docs)

    print(f"\nTotal chunks: {len(chunks)}")
    print("\n--- Sample code chunk ---")
    code = next((c for c in chunks if c.metadata["type"] == "code"), None)
    if code:
        print(f"Source: {code.metadata['source']}")
        print(f"Chunk index: {code.metadata['chunk_index']}")
        print(f"Content:\n{code.page_content[:300]}")

    print("\n--- Sample issue chunk ---")
    issue = next((c for c in chunks if c.metadata["type"] == "issue"), None)
    if issue:
        print(f"Title: {issue.metadata['title']}")
        print(f"Assignees: {issue.metadata['assignees']}")
        print(f"Content:\n{issue.page_content[:300]}")