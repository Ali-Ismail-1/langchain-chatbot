from __future__ import annotations
import os
from typing import Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

from app.settings import settings

# ---------- Build / load vector store ----------
def build_or_load_vectorstore(doc_dir: str, chroma_dir: str) -> Chroma:
    os.makedirs(chroma_dir, exist_ok=True)

    # Load docs (simple: all .txt/.md in folder)
    docs = []
    for root, _, files in os.walk(doc_dir):
        for f in files:
            if f.endswith((".txt", ".md")):
                path = os.path.join(root, f)
                docs.extend(TextLoader(path, encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)

     # New: always create/load the DB first
    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model)
    vectordb = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

    # Use persisted Chroma
    if splits:
        vectordb.add_documents(splits)
        vectordb.persist()
    return vectordb

# Initialize once at import
VDB = build_or_load_vectorstore("app/data/sample_docs", settings.chroma_dir)
RETRIEVER = VDB.as_retriever(search_kwargs={"k": 4})

# ---------- LLM ----------
def get_llm():
    if settings.llm_provider == "openai":
        return ChatOpenAI(model=settings.openai_model, temperature=0.1, api_key=settings.openai_api_key)
    raise ValueError("Unsupported LLM provider (set LLM_PROVIDER=openai or add another provider).")

# ---------- Prompt & chain ----------
BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use the provided context to answer. "
     "If the answer is not in the context, say you don't know."),
    MessagesPlaceholder("history"),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def make_rag_chain():
    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

    # RAG: retrieve -> prompt -> LLM
    def chain_fn(inputs, config):
        question = inputs["question"]
        docs = RETRIEVER.invoke(question)
        ctx = format_docs(docs)
        prompt = BASE_PROMPT.format(
            history=config.get("history", []),
            question=question,
            context=ctx
        )
        return llm.invoke(prompt)

    # Wrap with message history per-session
    store: Dict[str, ChatMessageHistory] = {}

    def get_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    runnable = RunnableWithMessageHistory(
        runnable=chain_fn,
        get_session_history=get_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    return runnable

RAG = make_rag_chain()

def ask_with_context(session_id: str, question: str) -> str:
    resp = RAG.invoke({"question": question}, config={"configurable": {"session_id": session_id}})
    # Normalize return type (AIMessage or str)
    if isinstance(resp, AIMessage):
        return resp.content
    return str(resp)
