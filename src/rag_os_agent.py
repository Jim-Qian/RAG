import os
import uuid
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

import chromadb
from chromadb.config import Settings

from pypdf import PdfReader

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Local Chroma DB
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"
    )
)
collection = chroma_client.get_or_create_collection("desktop_docs")


# ----------------------------------------------------------------------
# Helpers: file reading + chunking
# ----------------------------------------------------------------------

def load_text_from_file(path: str) -> str:
    """Extract text from txt/md/pdf."""
    if path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            print(f"Failed PDF read {path}: {e}")
            return ""
    else:
        # txt, md, etc.
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except:
            print(f"Failed text read {path}")
            return ""


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Simple character-based chunking."""
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap

    return chunks


# ----------------------------------------------------------------------
# Embedding helper
# ----------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch embedding request."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


# ----------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------

def index_directory(root_dir: str):
    """
    Build the vector index: read files → chunk → embed → store in Chroma.
    """
    docs = []
    metadatas = []
    ids = []

    print(f"\nIndexing directory: {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not any(fname.lower().endswith(ext) for ext in [".txt", ".md", ".pdf"]):
                continue

            full_path = os.path.join(dirpath, fname)
            print(f"  Reading {full_path}")

            text = load_text_from_file(full_path)
            if not text:
                continue

            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                docs.append(chunk)
                metadatas.append({
                    "source_path": full_path,
                    "chunk_index": idx,
                })
                ids.append(str(uuid.uuid4()))

    if not docs:
        print("No documents found.")
        return

    print("Embedding chunks...")
    embeddings = embed_texts(docs)

    print("Adding to Chroma...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=docs,
        metadatas=metadatas,
    )
    chroma_client.persist()

    print(f"Done. Indexed {len(docs)} chunks.\n")


# ----------------------------------------------------------------------
# RAG: retrieve & answer
# ----------------------------------------------------------------------

def retrieve_chunks(query: str, k: int = 5) -> List[Dict]:
    """Embed the question → search local DB → return top chunks."""
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    chunks = []
    for doc, meta in zip(docs, metas):
        chunks.append({
            "text": doc,
            "source_path": meta["source_path"],
            "chunk_index": meta["chunk_index"],
        })

    return chunks


def build_prompt(question: str, chunks: List[Dict]) -> List[Dict]:
    """Construct chat prompt using retrieved chunks."""
    context = ""

    for i, ch in enumerate(chunks):
        context += (
            f"[Chunk {i+1} | {ch['source_path']} | part {ch['chunk_index']}]\n"
            f"{ch['text']}\n\n"
        )

    return [
        {
            "role": "system",
            "content": (
                "You are my personal OS assistant. "
                "Use the provided document context to answer the question. "
                "If context is insufficient, say so."
            )
        },
        {
            "role": "system",
            "content": f"CONTEXT:\n{context}"
        },
        {
            "role": "user",
            "content": question
        }
    ]


def answer_question(question: str) -> str:
    """Full RAG retrieval + OpenAI answer."""
    chunks = retrieve_chunks(question, k=5)
    if not chunks:
        return "No relevant documents found."

    messages = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )

    return response.choices[0].message.content


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        action="store_true",
        help="Rebuild the document index from ./data"
    )
    args = parser.parse_args()

    if args.index:
        index_directory("./data")

    print("📁 Personal RAG Desktop Agent ready.")
    print("Ask a question (or type 'exit').")

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        print("Thinking...\n")
        answer = answer_question(q)
        print("Assistant:", answer)
