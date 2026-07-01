import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"


def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=GROQ_MODEL,
        temperature=0.2
    )


def build_rag_chain(retriever):
    llm = get_llm()

    # Step 1 — contextualize the user question using chat history
    # if user says "explain it more" — this rephrases it to a standalone question
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the chat history and the latest user question, 
        reformulate the question to be standalone and clear.
        Do NOT answer it. Just reformulate if needed, else return as is."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Step 2 — answer generation prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert code assistant helping users understand GitHub repositories.
        
Use the retrieved context below to answer the user's question accurately.
- For code questions: explain what the code does, how it works, key functions/classes involved.
- For issue questions: summarize the issue, its labels, and whether it is assigned or not.
- For PR questions: explain what the PR changes and who authored it.
- For tech stack questions: list ALL technologies found — languages, frameworks, 
  libraries, databases, deployment tools. Check package.json, requirements.txt, 
  and README for this information.
- For architecture questions: describe the full system — frontend, backend, database, ML components.
- For property/field/list questions: if the context contains a class, schema, or structured list 
  defining multiple items (e.g. a Pydantic model, database schema, enum, or config object), you MUST 
  include EVERY item from that structure in your answer. Before finalizing your answer, count the 
  number of fields/items visible in the context and verify your answer mentions that same count.
- If the answer is partially in the context, use ONLY what's explicitly stated and clearly say what's missing.
- NEVER speculate, guess, or infer technical details (platforms, tools, architecture choices) that are not explicitly written in the context, even if they seem technically plausible.
- If you catch yourself about to say "likely" or "probably" about a technical fact, STOP and instead say the context doesn't specify that detail.
- Do NOT hallucinate. Stick strictly to the context.

Context:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Step 3 — stuff documents into the prompt and generate answer
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Step 4 — full chain: history-aware retrieval + answer generation
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    return rag_chain


def run_query(rag_chain, question: str, chat_history: list) -> dict:
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    return {
        "answer": response["answer"],
        "sources": list(set([
            doc.metadata.get("source", "unknown")
            for doc in response["context"]
        ]))
    }


def format_chat_history(history: list[dict]) -> list:
    # converts [{"role": "user/assistant", "content": "..."}]
    # into LangChain message objects
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


# Quick test
# if __name__ == "__main__":
#     from github_ingestor import ingest_repo
#     from chunker import chunk_documents
#     from vector_store import build_vector_store, get_retriever

#     raw_docs = ingest_repo("https://github.com/Asifmd45/Advanced-RAG-Pipeline")
#     chunks = chunk_documents(raw_docs)
#     vs = build_vector_store(chunks, "Asifmd45/Advanced-RAG-Pipeline")
#     retriever = get_retriever(vs)
#     chain = build_rag_chain(retriever)

#     chat_history = []

#     q1 = "What does this project do?"
#     result1 = run_query(chain, q1, format_chat_history(chat_history))
#     print(f"\nQ: {q1}")
#     print(f"A: {result1['answer']}")
#     print(f"Sources: {result1['sources']}")

#     chat_history.append({"role": "user", "content": q1})
#     chat_history.append({"role": "assistant", "content": result1["answer"]})

#     q2 = "What retrieval methods does it implement?"
#     result2 = run_query(chain, q2, format_chat_history(chat_history))
#     print(f"\nQ: {q2}")
#     print(f"A: {result2['answer']}")
#     print(f"Sources: {result2['sources']}")

if __name__ == "__main__":
    from github_ingestor import ingest_repo
    from chunker import chunk_documents
    from vector_store import build_vector_store, get_retriever

    raw_docs = ingest_repo("https://github.com/Asifmd45/LLM-PROP-REAL")
    chunks = chunk_documents(raw_docs)
    vs = build_vector_store(chunks, "Asifmd45/LLM-PROP-REAL")
    retriever = get_retriever(vs)
    chain = build_rag_chain(retriever)

    chat_history = []

    q1 = "What does this project do?"
    result1 = run_query(chain, q1, format_chat_history(chat_history))
    print(f"\nQ: {q1}")
    print(f"A: {result1['answer']}")
    print(f"Sources: {result1['sources']}")

    chat_history.append({"role": "user", "content": q1})
    chat_history.append({"role": "assistant", "content": result1["answer"]})

    q2 = "What tech stack is used in this project?"
    result2 = run_query(chain, q2, format_chat_history(chat_history))
    print(f"\nQ: {q2}")
    print(f"A: {result2['answer']}")
    print(f"Sources: {result2['sources']}")

    chat_history.append({"role": "user", "content": q2})
    chat_history.append({"role": "assistant", "content": result2["answer"]})

    q3 = "How are the T5 models used here?"
    result3 = run_query(chain, q3, format_chat_history(chat_history))
    print(f"\nQ: {q3}")
    print(f"A: {result3['answer']}")
    print(f"Sources: {result3['sources']}")