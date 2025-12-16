import os
import uuid
from typing import List, Dict, cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

import chromadb
from chromadb.api.types import Embedding, Metadata
from pypdf import PdfReader

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env")

client = OpenAI(api_key=api_key)

# Use a persistent Chroma client (data auto-saved under ./chroma_db)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("desktop_docs")


# ----------------------------------------------------------------------
# Helpers: file reading + chunking
# ----------------------------------------------------------------------

def load_text_from_file(path: str) -> str:
    """Extract text from txt/md/pdf."""
    path_lower = path.lower()

    if path_lower.endswith(".pdf"):
        try:
            reader = PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            print(f"Failed to read PDF {path}: {e}")
            return ""

    # Fallback: assume text-like file
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Failed to read text file {path}: {e}")
        return ""


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Simple character-based chunking."""
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        # avoid infinite loop when text is shorter than max_chars
        if end == n:
            break
        start = end - overlap

    return chunks


# ----------------------------------------------------------------------
# Embedding helper (OpenAI embeddings)
# ----------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[Embedding]:
    """Batch embedding request using OpenAI, returning Chroma-compatible Embeddings."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    embeddings: List[Embedding] = cast(
        List[Embedding],
        [d.embedding for d in response.data],
    )
    return embeddings


# ----------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------

def index_directory(root_dir: str) -> None:
    """
    Build the vector index: read files → chunk → embed → store in Chroma.
    """
    docs: List[str] = []
    metadatas: List[Metadata] = []
    ids: List[str] = []

    print(f"\nIndexing directory: {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not any(fname.lower().endswith(ext) for ext in [".txt", ".md", ".pdf"]):
                continue

            full_path = os.path.join(dirpath, fname)
            print(f"  Reading {full_path}")

            text = load_text_from_file(full_path)
            if not text.strip():
                continue

            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                docs.append(chunk)
                metadatas.append(
                    cast(
                        Metadata,
                        {
                            "source_path": full_path,
                            "chunk_index": idx,
                        },
                    )
                )
                ids.append(str(uuid.uuid4()))

    if not docs:
        print("No documents found to index.")
        return

    print(f"Embedding {len(docs)} chunks...")
    embeddings = embed_texts(docs)

    print("Adding to Chroma collection...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=docs,
        metadatas=metadatas,
    )

    print(f"Done. Indexed {len(docs)} chunks.\n")


# ----------------------------------------------------------------------
# RAG: retrieve & answer
# ----------------------------------------------------------------------

def retrieve_chunks(query: str, k: int = 5) -> List[Dict]:
    """Embed the question → search local DB → return top chunks."""
    query_embedding_list = embed_texts([query])
    if not query_embedding_list:
        return []

    query_embedding = query_embedding_list[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"],
    )

    documents = results.get("documents")
    metadatas = results.get("metadatas")

    # Guard against None or empty results
    if (
        not documents
        or not metadatas
        or len(documents) == 0
        or len(metadatas) == 0
        or documents[0] is None
        or metadatas[0] is None
    ):
        return []

    docs0 = documents[0]
    metas0 = metadatas[0]

    chunks: List[Dict] = []
    for doc, meta in zip(docs0, metas0):
        chunks.append(
            {
                "text": doc,
                "source_path": meta.get("source_path"),
                "chunk_index": meta.get("chunk_index"),
            }
        )

    return chunks


def build_prompt(question: str, chunks: List[Dict]) -> List[ChatCompletionMessageParam]:
    """Construct chat prompt using retrieved chunks."""
    context_parts: List[str] = []

    for i, ch in enumerate(chunks):
        context_parts.append(
            f"[Chunk {i+1} | {ch['source_path']} | part {ch['chunk_index']}]\n"
            f"{ch['text']}\n"
        )

    context_text = "\n".join(context_parts)

    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are my personal OS assistant. "
                "Use the provided document context to answer the question. "
                "If the context is insufficient, say that you are not sure."
            ),
        },
        {
            "role": "system",
            "content": f"CONTEXT:\n{context_text}",
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    return messages


def answer_question(question: str) -> str:
    """Full RAG retrieval + OpenAI answer."""
    chunks = retrieve_chunks(question, k=5)
    if not chunks:
        return "No relevant documents found in the index."

    messages = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )

    content = response.choices[0].message.content or ""
    return content


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        action="store_true",
        help="Rebuild the document index from ./data",
    )
    args = parser.parse_args()

    if args.index:
        index_directory("./data")

    print("📁 Personal RAG Desktop Agent ready.")
    print("Put files under ./data, then ask questions (or type 'exit').")

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        print("Thinking...\n")
        answer = answer_question(q)
        print("Assistant:", answer)
