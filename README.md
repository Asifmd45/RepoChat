# 🐙 RepoChat — Chat with Any GitHub Repository

RepoChat is an AI-powered **Retrieval-Augmented Generation (RAG)** application that lets you chat with any **public GitHub repository** using natural language.

Instead of manually browsing source code, issues, and pull requests, simply ask questions like:

* *"What does this project do?"*
* *"How does authentication work?"*
* *"Show me unassigned issues."*
* *"Explain the retrieval pipeline."*

RepoChat retrieves the most relevant repository content using semantic search and generates accurate, context-aware answers powered by Groq LLMs.

---

# 🎯 Use Cases

* 🚀 **New Contributors** — Understand an unfamiliar codebase quickly.
* 🛠️ **Open Source Contributors** — Discover issues, explore PRs, and find good first contributions.
* 📚 **Developers Evaluating Libraries** — Learn a project without reading every file manually.
* 🤖 **AI-Assisted Code Exploration** — Ask questions about architecture, functions, and implementation details.

---

# ✨ Features

* 🔍 **Codebase Understanding**

  * Explain project architecture
  * Understand files, classes, and functions
  * Explore technologies used

* 🐛 **Issue Exploration**

  * Search open issues
  * Filter by assignee or label
  * Discover good first issues

* 🔀 **Pull Request Navigation**

  * Summarize PRs
  * Understand changes
  * Identify contributors

* 💬 **Conversational Memory**

  * Supports follow-up questions
  * Maintains chat history automatically

* 🎯 **Context-Aware Retrieval**

  * Search across:

    * Code
    * Issues
    * Pull Requests
  * Filter retrieval by document type

* ⚡ **Fast Inference**

  * Powered by Groq's ultra-fast inference engine

---

# 🏗️ Architecture

```text
                 GitHub Repository URL
                          │
                          ▼
                GitHub API (PyGithub)
                          │
                          ▼
        Files + Issues + Pull Requests
                          │
                          ▼
     Code-Aware Document Chunking (LangChain)
                          │
                          ▼
       HuggingFace MiniLM Embeddings
                          │
                          ▼
         ChromaDB Vector Database
     (Cosine Similarity + MMR Retrieval)
                          │
                          ▼
        LangChain LCEL Retrieval Chain
      (History-Aware Conversational RAG)
                          │
                          ▼
          Groq LLM (Llama 3.1 Instant)
                          │
                          ▼
             Streamlit Chat Interface
```

---

# 🛠️ Tech Stack

| Layer              | Technology                              |
| ------------------ | --------------------------------------- |
| Frontend           | Streamlit                               |
| LLM                | Groq API (Llama 3.1 Instant)            |
| Embeddings         | HuggingFace MiniLM (`all-MiniLM-L6-v2`) |
| Vector Store       | ChromaDB                                |
| Retrieval          | Cosine Similarity + MMR                 |
| Orchestration      | LangChain LCEL                          |
| GitHub Integration | PyGithub                                |
| Language           | Python 3.10+                            |

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/Asifmd45/github-repo-rag.git
cd github-repo-rag
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
GITHUB_TOKEN=your_github_personal_access_token
GROQ_API_KEY=your_groq_api_key
```

### GitHub Personal Access Token

Generate one from:

```
https://github.com/settings/tokens
```

Recommended permission:

* `repo`

### Groq API Key

Generate one from:

```
https://console.groq.com
```

---

## 5. Launch the Application

```bash
streamlit run app.py
```

---

# 💬 Example Questions

## Codebase

* What does this project do?
* Explain the overall architecture.
* How does authentication work?
* Explain the main entry point.
* Which embedding model is used?
* How does retrieval work?

## Issues

* Show me open issues.
* Show me unassigned issues.
* Are there any good first issues?
* What bugs are currently reported?

## Pull Requests

* What PRs are currently open?
* Who contributed recently?
* Summarize the latest PR.

---

# 📁 Project Structure

```text
github-repo-rag/
│
├── app.py
├── github_ingestor.py
├── chunker.py
├── vector_store.py
├── rag_chain.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

| File                 | Purpose                                             |
| -------------------- | --------------------------------------------------- |
| `app.py`             | Streamlit user interface                            |
| `github_ingestor.py` | Fetches repository files, issues, and pull requests |
| `chunker.py`         | Splits repository content into semantic chunks      |
| `vector_store.py`    | Creates and loads the ChromaDB vector database      |
| `rag_chain.py`       | Builds the conversational RAG pipeline              |

---

# 🔑 How It Works

### 1. Repository Ingestion

PyGithub downloads:

* Repository files
* Open issues
* Pull requests

---

### 2. Chunking

Documents are split using type-aware strategies:

* Code → logical code chunks
* Markdown → section-based chunks
* Issues & PRs → structured text chunks

---

### 3. Embedding

Each chunk is converted into vector embeddings using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Metadata such as:

* source
* document type
* issue number
* assignee

is stored alongside each embedding.

---

### 4. Retrieval

The retriever performs:

* Semantic similarity search
* Maximum Marginal Relevance (MMR)

This retrieves diverse and relevant context for each query.

---

### 5. Response Generation

LangChain:

1. Reformulates follow-up questions using chat history.
2. Retrieves relevant documents.
3. Injects retrieved context into the prompt.
4. Generates an answer using Groq LLM.
5. Returns the response along with source references.

---

# ⚠️ Limitations

* Supports **public GitHub repositories only**
* Large repositories (1000+ files) may require several minutes for indexing
* Response quality depends on retrieved context
* Subject to Groq API rate limits

---

# 👤 Author

**Asif MD**

GitHub: https://github.com/Asifmd45

---

## ⭐ If you found this project useful, consider giving it a star!
