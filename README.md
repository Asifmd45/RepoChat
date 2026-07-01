# 🐙 RepoChat — Chat with Any GitHub Repository

RepoChat is an AI-powered RAG (Retrieval-Augmented Generation) application that lets you have a conversation with any public GitHub repository. Understand codebases instantly, explore open issues, and navigate pull requests — all through natural language.

---

## 🎯 Use Cases

- **New contributors** — Understand an unfamiliar codebase without reading every file
- **Open source contributors** — Find unassigned issues, explore PRs, identify where to start
- **Developers evaluating libraries** — Ask questions instead of digging through docs

---

## ✨ Features

- 🔍 **Codebase understanding** — Ask about architecture, specific files, functions, and tech stack
- 🐛 **Issue exploration** — Query open issues, filter by assignee or label
- 🔀 **PR navigation** — Understand what each PR changes and who authored it
- 💬 **Conversational memory** — Follow-up questions maintain full chat context
- 🎯 **Scope filtering** — Search across all content or filter to code / issues / PRs only
- ⚡ **Fast inference** — Powered by Groq's LPU for near-instant responses
- 📦 **Smart caching** — Already-loaded repos reload instantly from disk
- 🚀 **Demo repos** — One-click load for micrograd, Flask, and Express

---

## 🏗️ Architecture

```
GitHub Repo URL
      ↓
GitHub API — get_git_tree() [1 API call for full tree]
      ↓
Parallel get_git_blob() + Issues + PRs [ThreadPoolExecutor, 20 workers]
      ↓
Files + Issues + PRs (raw documents with metadata)
      ↓
Code-aware Chunker (RecursiveCharacterTextSplitter)
      ↓
MiniLM Embeddings (HuggingFace)
      ↓
ChromaDB Vector Store (cosine similarity, MMR retrieval)
      ↓
LangChain LCEL Chain (history-aware retriever + stuff documents)
      ↓
Groq LLM (llama-3.1-8b-instant)
      ↓
Streamlit Chat UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq API — llama-3.1-8b-instant |
| Embeddings | HuggingFace MiniLM (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (MMR retrieval, cosine similarity) |
| Orchestration | LangChain LCEL |
| GitHub Integration | PyGithub |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Asifmd45/RepoChat.git
cd RepoChat
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root:
```
GITHUB_TOKEN=your_github_personal_access_token
GROQ_API_KEY=your_groq_api_key
```

- GitHub token: [github.com/settings/tokens](https://github.com/settings/tokens) — select `repo` scope
- Groq API key: [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

---

## 💬 Example Questions

**Codebase:**
- `What does this project do?`
- `What is the overall architecture?`
- `How does authentication work?`
- `Explain the main entry point`

**Issues:**
- `Show me unassigned open issues`
- `Any good first issues available?`
- `What bugs are currently reported?`

**Pull Requests:**
- `What PRs are currently open?`
- `Who is actively contributing?`

---

## 📁 Project Structure

```
RepoChat/
├── app.py                  # Streamlit UI
├── github_ingestor.py      # GitHub API — tree + blob fetching, issues, PRs
├── chunker.py              # Code-aware document chunking
├── vector_store.py         # ChromaDB setup and MMR retrieval
├── rag_chain.py            # LangChain LCEL chain with Groq
├── .env                    # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔑 How It Works

1. **Size check** — Tree API call (1 second) counts valid files upfront; repos over 600 files are flagged before ingestion starts
2. **Ingestion** — `get_git_tree(recursive=True)` fetches the entire file tree in one API call; file content fetched in parallel via `get_git_blob(sha)`
3. **Chunking** — Type-aware strategies: code splits at function boundaries, markdown at headers, issues stay whole
4. **Embedding** — Each chunk embedded with MiniLM and stored in ChromaDB with metadata (type, source, issue number, assignee)
5. **Caching** — Already-indexed repos load from disk instantly on repeat visits
6. **Retrieval** — MMR fetches 20 candidates, returns 8 diverse relevant chunks
7. **Generation** — History-aware chain reformulates follow-up questions; Groq LLM generates grounded answers with sources

---

## ⚠️ Limitations

- Only works with **public** GitHub repositories
- Repos over **600 files** are blocked upfront (GitHub API rate limit protection)
- Groq free tier: 6,000 tokens/minute, 500,000 tokens/day
- Only **open** issues and PRs are indexed (closed/merged not included)

---

## 👤 Author

**Asif MD** — [github.com/Asifmd45](https://github.com/Asifmd45)