import streamlit as st
from github_ingestor import ingest_repo
from chunker import chunk_documents
from vector_store import build_vector_store, load_vector_store, get_retriever
from rag_chain import build_rag_chain, run_query, format_chat_history
import os

st.set_page_config(
    page_title="RepoChat",
    page_icon="🐙",
    layout="wide"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🐙 RepoChat")
    st.caption("Chat with any GitHub repository")
    st.divider()

    repo_url = st.text_input(
        "GitHub Repo URL",
        placeholder="https://github.com/owner/repo",
        key="repo_url_input"
    )

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

    if "repo_loaded" in st.session_state and st.session_state.repo_loaded:
        st.success(f"✅ Loaded")
        st.caption(st.session_state.loaded_repo_url)
        st.caption(f"Chunks indexed: {st.session_state.chunk_count}")

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq · MiniLM")


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


# ── Repo loading ──────────────────────────────────────────────────────────────
if load_btn and repo_url.strip():
    with st.spinner("Fetching repo files, issues & PRs..."):
        try:
            raw_docs = ingest_repo(repo_url.strip())
            chunks = chunk_documents(raw_docs)

            repo_name = repo_url.strip().rstrip("/")
            repo_name = "/".join(repo_name.replace("https://", "").replace("http://", "").split("/")[1:3])

            vs = build_vector_store(chunks, repo_name)

            doc_type_map = {
                "All (code + issues + PRs)": None,
                "Code only": "code",
                "Issues only": "issue",
                "Pull Requests only": "pull_request"
            }
            doc_type = doc_type_map[filter_option]
            retriever = get_retriever(vs, doc_type)
            chain = build_rag_chain(retriever)

            st.session_state.rag_chain = chain
            st.session_state.repo_loaded = True
            st.session_state.loaded_repo_url = repo_url.strip()
            st.session_state.chunk_count = len(chunks)
            st.session_state.chat_history = []

            st.rerun()

        except Exception as e:
            st.error(f"Failed to load repo: {str(e)}")


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
    # render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.caption(src)

    # chat input
    user_input = st.chat_input("Ask anything about this repository...")

    if user_input:
        # show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_query(
                    st.session_state.rag_chain,
                    user_input,
                    format_chat_history(st.session_state.chat_history)
                )
                st.markdown(result["answer"])
                with st.expander("📎 Sources"):
                    for src in result["sources"]:
                        st.caption(src)

        # update history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })