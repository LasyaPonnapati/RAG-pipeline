import hashlib
import os
import shutil
import time
from pathlib import Path

import cohere
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI

load_dotenv()

CHUNKS_PATH = Path(__file__).with_name("hybrid_search_chunks.txt")
PERSIST_DIRECTORY = "hybrid-search-db"
QUERY_VARIANTS = 5
RETRIEVAL_K = 10
EMBED_BATCH_SIZE = 80
EMBED_BATCH_PAUSE_SECONDS = 65
GROQ_MODEL = "openai/gpt-oss-20b"
COHERE_RERANK_MODEL = "rerank-v3.5"
RERANK_TOP_N = 5

ANSWER_PROMPT = (
    "You are a helpful assistant that answers questions about the documents provided. "
    "Use only those documents. If they are not enough, say you don't know. "
    "Return the answer in a concise and clear manner. "
    "When the documents describe separate steps or events, keep them separate. "
    "Do not merge them into one step. "
    "If a document lists repeated actions in order, such as "
    "'once to A, once to B, and once for C', that list is the order of those actions. "
    "Do not add an action before that list. "
    "Do not describe an action as launch or upward acceleration unless a document says that. "
    "Match each named event to the list item with the same description. "
    "Do not move a description from one list item onto another. "
    "Place any other event using only the time words in its own document, "
    "such as 'before', 'after', or 'in the final seconds'. "
    "Document numbers are relevance rank, not time order."
)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MULTI_QUERY_PROMPT = (
    "Rewrite the user question into 5 different phrasings that could retrieve "
    "different relevant passages. Keep the same meaning. "
    "Do not answer the question. Return exactly 5 questions, one per line, "
    "with no numbering or extra text."
)


def load_chunks(path: Path = CHUNKS_PATH) -> str:
    return path.read_text(encoding="utf-8")


def chunks_to_documents(text: str) -> list[Document]:
    sentences = [part.strip() for part in text.split("\n\n") if part.strip()]
    return [Document(page_content=sentence) for sentence in sentences]


def create_vector_store(documents: list[Document], persist_directory: str) -> Chroma:
    print(f"Creating vector store in {persist_directory}...")

    os.makedirs(persist_directory, exist_ok=True)

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    # Free tier allows 100 embed requests per minute, so add in smaller batches.
    for start in range(0, len(documents), EMBED_BATCH_SIZE):
        if start:
            print(
                f"Pausing {EMBED_BATCH_PAUSE_SECONDS}s to stay under the embedding quota..."
            )
            time.sleep(EMBED_BATCH_PAUSE_SECONDS)
        batch = documents[start : start + EMBED_BATCH_SIZE]
        end = start + len(batch)
        print(f"Embedding chunks {start + 1}-{end} of {len(documents)}...")
        db.add_documents(batch)
    return db


def load_vector_store(persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )


def print_docs(title: str, docs: list[Document]) -> None:
    print(f"\n=== {title} ({len(docs)} docs) ===")
    if not docs:
        print("No documents matched.")
        return
    for i, doc in enumerate(docs, start=1):
        print(f"\n[Chunk {i}] {doc.page_content}")


def similarity_search(db: Chroma, query: str, k: int = 3) -> list[Document]:
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def bm25_search(documents: list[Document], query: str, k: int = 3) -> list[Document]:
    retriever = BM25Retriever.from_documents(documents, k=k)
    return retriever.invoke(query)


def ensemble_search(
    db: Chroma, documents: list[Document], query: str, k: int = 3
) -> list[Document]:
    similarity_retriever = db.as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)
    # Weighted reciprocal rank fusion: score += weight / (rank + 60).
    # Uses each list's rank, not the raw similarity or BM25 score.
    retriever = EnsembleRetriever(
        retrievers=[similarity_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    return retriever.invoke(query)[:k]


def generate_query_variants(query: str, n: int = QUERY_VARIANTS) -> list[str]:
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing from the environment (.env)")

    print(f"Asking {GROQ_MODEL} for {n} phrasings...")
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": MULTI_QUERY_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    content = response.choices[0].message.content or ""
    variants = []
    for line in content.strip().splitlines():
        text = line.strip().lstrip("0123456789.-) ").strip()
        if text:
            variants.append(text)
    return variants[:n]


def get_vector_store(documents: list[Document], source_text: str) -> Chroma:
    fingerprint = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    fingerprint_path = Path(PERSIST_DIRECTORY) / "corpus.sha256"
    if fingerprint_path.is_file() and fingerprint_path.read_text(encoding="utf-8") == fingerprint:
        print(f"Loading existing vector store from {PERSIST_DIRECTORY}...")
        return load_vector_store()

    if os.path.isdir(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    db = create_vector_store(documents, PERSIST_DIRECTORY)
    fingerprint_path.write_text(fingerprint, encoding="utf-8")
    return db


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], top_n: int = 10
) -> list[tuple[Document, float]]:
    # Same fusion as the ensemble retriever: score += 1 / (rank + 60).
    scores: dict[str, float] = {}
    best_doc: dict[str, Document] = {}
    for docs in ranked_lists:
        for position, doc in enumerate(docs, start=1):
            text = doc.page_content
            scores[text] = scores.get(text, 0.0) + 1 / (60 + position)
            best_doc.setdefault(text, doc)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [(best_doc[text], score) for text, score in ranked]


def cohere_rerank(
    query: str, docs: list[Document], top_n: int = RERANK_TOP_N
) -> list[tuple[Document, float]]:
    if not os.getenv("COHERE_API_KEY"):
        raise ValueError("COHERE_API_KEY is missing from the environment (.env)")
    if not docs:
        return []

    cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    response = cohere_client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=[doc.page_content for doc in docs],
        top_n=top_n,
    )
    return [(docs[result.index], result.relevance_score) for result in response.results]


def generate_answer(query: str, docs: list[Document]) -> str:
    context = "\n\n".join(
        f"[Doc {i}]\n{doc.page_content}" for i, doc in enumerate(docs, start=1)
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": ANSWER_PROMPT},
            {
                "role": "user",
                "content": (
                    "Use only the following context to answer the question. "
                    "If the context is not sufficient, say you don't know. "
                    "Keep each described action separate, and keep the order a document lists.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {query}"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def multi_query_retrieval(
    db: Chroma,
    documents: list[Document],
    query: str,
    k: int = RETRIEVAL_K,
) -> None:
    variants = generate_query_variants(query)
    print(f"\nUser question: {query}")
    print(f"Generated {len(variants)} phrasings.")
    if len(variants) < QUERY_VARIANTS:
        print(f"Expected {QUERY_VARIANTS} phrasings and got {len(variants)}.")
    if not variants:
        print("No query variants were generated.")
        return

    for i, question in enumerate(variants, start=1):
        print(f"{i}. {question}")

    ensemble_lists = [
        ensemble_search(db, documents, question, k=k) for question in variants
    ]
    fused = reciprocal_rank_fusion(ensemble_lists, top_n=10)

    if not fused:
        print("No documents matched.")
        return

    reranked = cohere_rerank(query, [doc for doc, _score in fused], top_n=RERANK_TOP_N)
    print(f"\n=== Cohere rerank (top {len(reranked)}) ===")
    if not reranked:
        print("No documents matched.")
        return
    for i, (doc, score) in enumerate(reranked, start=1):
        print(f"\n[Chunk {i}] relevance={score:.4f}")
        print(doc.page_content)

    answer = generate_answer(query, [doc for doc, _score in reranked])
    print("\n=== Answer ===")
    print(answer)


def main():
    query = input("Enter your query: ").strip()
    if not query:
        print("No query entered.")
        return

    source_text = load_chunks()
    documents = chunks_to_documents(source_text)
    print(f"Loaded {len(documents)} chunks from {CHUNKS_PATH.name}.")
    db = get_vector_store(documents, source_text)
    multi_query_retrieval(db, documents, query)


if __name__ == "__main__":
    main()
