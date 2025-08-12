# app / rag.py
from __future__ import annotations
import os
from typing import Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOllama

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

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
    return vectordb

# Initialize once at import
VDB = build_or_load_vectorstore("app/data/sample_docs", settings.chroma_dir)
RETRIEVER = VDB.as_retriever(search_kwargs={"k": 4})


# ---------- LLM ----------
def get_llm():
    if settings.llm_provider == "ollama":
        kwargs = {}
        if settings.ollama_base_url:
            kwargs["base_url"] = settings.ollama_base_url
        return ChatOllama(model=settings.ollama_model, temperature=0.1, **kwargs)
    
    if settings.llm_provider == "openai":
        return ChatOpenAI(model=settings.openai_model, temperature=0.1, api_key=settings.openai_api_key)
    
    raise ValueError("Unsupported LLM provider (set LLM_PROVIDER=openai or add another provider).")

# ---------- Prompt & chain ----------
BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use the provided context to answer. "
     "Use context for factual/company questions; use chat history for user-specific preferences.\n"
     "If the information is in neither, say you don't know."),
    MessagesPlaceholder("history"),
    ("human", "Question: {input}\n\nContext:\n{context}")
])

def make_rag_chain():
    llm = get_llm()
    doc_chain = create_stuff_documents_chain(llm, BASE_PROMPT)
    retrieval_chain = create_retrieval_chain(RETRIEVER, doc_chain)

    store: dict[str, ChatMessageHistory] = {}

    def get_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    runnable = RunnableWithMessageHistory(
        runnable=retrieval_chain,
        get_session_history=get_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    return runnable

RAG = make_rag_chain()

def ask_with_context(session_id: str, question: str) -> str:
    result = RAG.invoke(
        {"input": question}, 
        config={"configurable": {"session_id": session_id}},
    )

    return result.get("answer") or str(result)
