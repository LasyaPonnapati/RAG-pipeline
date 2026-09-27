import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

chunks = '''
SpaceX's Raptor engine burns liquid methane and oxygen at very high pressure, which gives Starship more thrust from a smaller engine.

Falcon 9's first stage fires its engines three times after launch so it can slow down and land upright on a pad or ship.

OpenAI models like GPT read the words so far and guess the next word using attention, which decides which earlier words matter most.

ChatGPT is tuned with human feedback: people rank sample answers, then training pushes the model toward the replies that were ranked higher.

Kubernetes stores the apps you want in a database called etcd, then starts or restarts containers until the cluster matches that list.

Each Kubernetes pod gets its own IP address, and a Service spreads incoming traffic across the healthy pods that are ready to take it.

A lithium-ion cell stores energy by moving lithium ions from a cathode into a graphite anode while a phone or car charges.

A battery management system watches each cell's voltage and temperature so one weak cell does not overcharge and damage the whole pack.

PostgreSQL uses a B-tree index to keep keys in sorted order, so finding a matching row takes only a few page reads.

PostgreSQL writes each change to a log before it updates the table, so after a crash it can replay that log and recover the data.
'''


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

    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vector_store(persist_directory: str = "hybrid-search-db") -> Chroma:
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


def main():
    query = input("Enter your query: ").strip()
    if not query:
        print("No query entered.")
        return

    documents = chunks_to_documents(chunks)
    db = load_vector_store()

    print_docs("Similarity search", similarity_search(db, query))
    print_docs("BM25 search", bm25_search(documents, query))
    print_docs("Ensemble search", ensemble_search(db, documents, query))


if __name__ == "__main__":
    main()
