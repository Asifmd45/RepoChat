import streamlit as st
from github_ingestor import ingest_repo, RepoTooLargeError
from chunker import chunk_documents
from vector_store import build_vector_store, load_vector_store, get_retriever
from rag_chain import build_rag_chain, run_query, format_chat_history
import os

st.set_page_config(
    page_title="RepoChat",
    page_icon="🐙",
    layout="wide"
)

DEMO_REPOS = [
    {
        "name": "micrograd",
        "repo_url": "https://github.com/karpathy/micrograd",
        "description": "Tiny autograd engine by Karpathy",
        "audience": "⚡ ML / Deep Learning",
        "questions": [
            "What does this project do?",
            "How does backpropagation work here?",
            "Explain the Value class and its role"
        ]
    },
    {
        "name": "flask",
        "repo_url": "https://github.com/pallets/flask",
        "description": "Python micro web framework",
        "audience": "🐍 Python / Web Dev",
        "questions": [
            "What is the overall architecture?",
            "How does request routing work?",
            "How does Flask handle middleware?"
        ]
    },
    {
        "name": "express",
        "repo_url": "https://github.com/expressjs/express",
        "description": "Fast Node.js web framework",
        "audience": "🟩 Node.js / Backend",
        "questions": [
            "What does this project do?",
            "How does middleware chaining work?",
            "How are routes defined and matched?"
        ]
    }
]

UNIVERSAL_QUESTIONS = [
    "What does this project do?",
    "What is the overall system architecture?",
    "What is the complete tech stack used?"
]


def estimate_load_time(doc_count: int) -> str:
    if doc_count < 100:
        return "10-20 seconds"
    elif doc_count < 500:
        return "30-60 seconds"
    elif doc_count < 1500:
        return "1-3 minutes"
    elif doc_count < 3000:
        return "3-6 minutes"
    else:
        return "6+ minutes (very large repo)"


def get_suggested_questions(repo_url: str) -> list[str]:
    for demo in DEMO_REPOS:
        if demo["repo_url"] == repo_url:
            return demo["questions"]
    return UNIVERSAL_QUESTIONS


def trigger_load(url: str):
    st.session_state.pending_load_url = url


# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "repo_loaded" not in st.session_state:
    st.session_state.repo_loaded = False
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "loaded_repo_url" not in st.session_state:
    st.session_state.loaded_repo_url = ""
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "demo_url" not in st.session_state:
    st.session_state.demo_url = None
if "pending_load_url" not in st.session_state:
    st.session_state.pending_load_url = None
if "questions_dismissed" not in st.session_state:
    st.session_state.questions_dismissed = False
if "prefill_url" not in st.session_state:
    st.session_state.prefill_url = ""


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🐙 RepoChat")
    st.caption("Chat with any GitHub repository")
    st.divider()

    repo_url = st.text_input(
        "GitHub Repo URL",
        value=st.session_state.prefill_url,
        placeholder="https://github.com/owner/repo",
        key="repo_url_input"
    )

    # demo pills
    st.caption("✨ Try a demo:")
    pill_cols = st.columns(3)
    for i, demo in enumerate(DEMO_REPOS):
        with pill_cols[i]:
            if st.button(
                demo["name"],
                key=f"pill_{demo['name']}",
                use_container_width=True
            ):
                st.session_state.prefill_url = demo["repo_url"]
                st.session_state.pending_load_url = demo["repo_url"]
                st.rerun()

    filter_option = st.selectbox(
        "Search scope",
        ["All (code + issues + PRs)", "Code only", "Issues only", "Pull Requests only"],
        key="filter_option"
    )

    load_btn = st.button("Load Repository", type="primary", use_container_width=True)

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.repo_loaded:
        st.success("✅ Loaded")
        st.caption(st.session_state.loaded_repo_url)
        st.caption(f"Chunks indexed: {st.session_state.chunk_count}")

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq · MiniLM")


# ── Determine URL to load ─────────────────────────────────────────────────────
url_to_load = None

if load_btn and repo_url.strip():
    url_to_load = repo_url.strip()
elif st.session_state.pending_load_url:
    url_to_load = st.session_state.pending_load_url
    st.session_state.pending_load_url = None


# ── Repo loading ──────────────────────────────────────────────────────────────
def load_repo(url: str):
    repo_name = url.strip().rstrip("/")
    repo_name = "/".join(repo_name.replace("https://", "").replace("http://", "").split("/")[1:3])
    collection_name = repo_name.replace("/", "_").replace("-", "_").lower()
    persist_dir = os.path.join("./chroma_db", collection_name)

    doc_type_map = {
        "All (code + issues + PRs)": None,
        "Code only": "code",
        "Issues only": "issue",
        "Pull Requests only": "pull_request"
    }
    doc_type = doc_type_map[filter_option]

    if os.path.exists(persist_dir):
        with st.status("Loading cached repository...", expanded=True) as status:
            st.write("📦 Found existing index, loading from cache...")
            vs = load_vector_store(repo_name)
            retriever = get_retriever(vs, doc_type)
            chain = build_rag_chain(retriever)
            st.session_state.rag_chain = chain
            st.session_state.vector_store = vs
            st.session_state.questions_dismissed = False
            st.session_state.repo_loaded = True
            st.session_state.loaded_repo_url = url
            st.session_state.chat_history = []
            st.session_state.prefill_url = ""
            status.update(label="✅ Repository loaded from cache!", state="complete")
        st.rerun()
    else:
        with st.status("Loading repository...", expanded=True) as status:
            try:
                st.write("🔍 Scanning repository tree...")
                raw_docs = ingest_repo(url)
                st.write(f"✅ Fetched {len(raw_docs)} documents (files, issues, PRs)")
                estimated = estimate_load_time(len(raw_docs))
                st.info(f"⏱️ Estimated remaining time: {estimated}")
                if len(raw_docs) > 1500:
                    st.warning(f"⚠️ Large repository ({len(raw_docs)} documents). This may take several minutes.")
                st.write("✂️ Splitting content into chunks...")
                chunks = chunk_documents(raw_docs)
                st.write(f"✅ Created {len(chunks)} chunks")
                st.write("🧠 Generating embeddings and building vector store...")
                vs = build_vector_store(chunks, repo_name)
                st.write("✅ Vector store ready")
                retriever = get_retriever(vs, doc_type)
                chain = build_rag_chain(retriever)
                st.session_state.rag_chain = chain
                st.session_state.vector_store = vs
                st.session_state.repo_loaded = True
                st.session_state.loaded_repo_url = url
                st.session_state.chunk_count = len(chunks)
                st.session_state.chat_history = []
                status.update(label="✅ Repository loaded successfully!", state="complete")
                st.rerun()
            except RepoTooLargeError as e:
                status.update(label="❌ Repository too large", state="error")
                st.error(str(e))
            except Exception as e:
                status.update(label="❌ Failed to load repository", state="error")
                st.error(f"Error: {str(e)}")


if url_to_load:
    load_repo(url_to_load)


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🐙 RepoChat")
st.caption("Understand any GitHub repository through conversation")

if not st.session_state.repo_loaded:
    st.info("👈 Paste a GitHub repo URL in the sidebar and click **Load Repository** to begin.")

    st.subheader("What you can ask:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📁 Codebase**")
        st.markdown("- What does this project do?")
        st.markdown("- How does authentication work?")
        st.markdown("- What is the folder structure?")
        st.markdown("- Explain the main entry point")
    with col2:
        st.markdown("**🐛 Issues**")
        st.markdown("- Show unassigned open issues")
        st.markdown("- Any good first issues?")
        st.markdown("- What bugs are reported?")
        st.markdown("- Issues related to login?")
    with col3:
        st.markdown("**🔀 Pull Requests**")
        st.markdown("- What PRs are open?")
        st.markdown("- What does PR #5 change?")
        st.markdown("- Who is actively contributing?")
        st.markdown("- Any PRs related to performance?")

else:
    # suggested questions at top of chat
    if not st.session_state.chat_history and not st.session_state.questions_dismissed:
        suggested = get_suggested_questions(st.session_state.loaded_repo_url)
        st.markdown("**💡 Suggested questions to get started:**")
        sq_cols = st.columns(3)
        for i, q in enumerate(suggested):
            with sq_cols[i]:
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.session_state.questions_dismissed = True
                    st.rerun()

    # render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.caption(src)

    # handle suggested question click
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    user_input = st.chat_input("Ask anything about this repository...")

    # use pending question if set, otherwise use chat input
    final_input = st.session_state.pending_question or user_input
    if st.session_state.pending_question:
        st.session_state.pending_question = None

    if final_input:
        with st.chat_message("user"):
            st.markdown(final_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_query(
                    st.session_state.rag_chain,
                    final_input,
                    format_chat_history(st.session_state.chat_history)
                )
                st.markdown(result["answer"])
                with st.expander("📎 Sources"):
                    for src in result["sources"]:
                        st.caption(src)

        st.session_state.chat_history.append({
            "role": "user",
            "content": final_input
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })