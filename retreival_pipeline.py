import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that can answer questions about the documents provided. "
    "You will be given a question and a list of documents. "
    "You will need to use the documents to answer the question. "
    "You will need to return the answer in a concise and clear manner. "
    "You will need to use markdown when it improves readability. "
    "For math, use LaTeX: \\(inline\\) and \\[block\\]. "
)

def retrieve(db: Chroma, query: str) -> list[Document]:
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)
    return relevant_docs

def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"[Doc {i} | {source}"
        if page is not None:
            label += f" p.{page}"
        label += "]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(parts)

def build_user_message(query: str, docs: list[Document]) -> str:
    context = format_docs(docs)
    return (
        "Use only the following context to answer the question. "
        "If the context is not sufficient, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

def main():
    query = input("Enter a question: ")

    # Same embedding model used during ingestion — required to query existing DB
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Opens the EXISTING db (does not create a new one)
    db = Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embedding_model,
    )

    relevant_docs = retrieve(db, query)
    user_message = build_user_message(query, relevant_docs)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
