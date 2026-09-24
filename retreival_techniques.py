import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MULTI_QUERY_PROMPT = (
    "Rewrite the user question into 3 different phrasings that could retrieve "
    "different relevant passages from a document. Keep the same meaning. "
    "Do not answer the question. Return exactly 3 questions, one per line, "
    "with no numbering or extra text."
)

def load_vector_store() -> Chroma:
    """Open the existing Chroma database."""
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    return Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embedding_model,
    )


def similarity_search(db: Chroma, query: str) -> None:
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)
    print(f"\n=== Similarity search ({len(relevant_docs)} docs) ===")
    if not relevant_docs:
        print("No documents matched.")
        return
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"\n[Chunk {i}] {doc.page_content}")


def similarity_search_with_threshold(db: Chroma, query: str) -> None:
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.7})
    relevant_docs = retriever.invoke(query)
    print(f"\n=== Similarity search with threshold ({len(relevant_docs)} docs) ===")
    if not relevant_docs:
        print("No documents matched.")
        return
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"\n[Chunk {i}] {doc.page_content}")


def max_marginal_relevance_search(db: Chroma, query: str) -> None:
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})
    relevant_docs = retriever.invoke(query)
    print(f"\n=== Maximum marginal relevance ({len(relevant_docs)} docs) ===")
    if not relevant_docs:
        print("No documents matched.")
        return
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"\n[Chunk {i}] {doc.page_content}")


def generate_query_variants(query: str) -> list[str]:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": MULTI_QUERY_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    lines = response.choices[0].message.content.strip().splitlines()
    variants = []
    for line in lines:
        text = line.strip().lstrip("0123456789.-) ").strip()
        if text:
            variants.append(text)
    return variants[:3]


def multi_query_retrieval(db: Chroma, query: str) -> None:
    variants = generate_query_variants(query)
    retreived_docs={question: [] for question in variants}
    print(f"\n=== Multi-query retrieval ({len(variants)} questions) ===")
    if not variants:
        print("No query variants were generated.")
        return
    for i, question in enumerate(variants, start=1):
        print(f"{i}. {question}")
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        retreived_docs[question].extend(docs)
    reciprocal_reranking(retreived_docs)

def reciprocal_reranking(retreived_docs: dict[str, list[Document]]) -> None:
    scores: dict[str, float] = {}
    for docs in retreived_docs.values():
        for position, doc in enumerate(docs, start=1):
            text = doc.page_content
            scores[text] = scores.get(text, 0.0) + 1 / (60 + position)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    print(f"\n=== Reciprocal rank fusion (top {len(ranked)} chunks) ===")
    if not ranked:
        print("No documents matched.")
        return
    for i, (text, score) in enumerate(ranked, start=1):
        print(f"\n[Chunk {i}] {text} score={score:.4f}")

def main():
    db = load_vector_store()
    query = input("Enter your query: ").strip()
    similarity_search(db, query)
    similarity_search_with_threshold(db, query)
    max_marginal_relevance_search(db, query)
    multi_query_retrieval(db, query)


if __name__ == "__main__":
    main()
