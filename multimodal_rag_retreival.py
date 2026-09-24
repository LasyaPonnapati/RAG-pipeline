import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI

load_dotenv()

MULTIMODAL_DB_DIR = "multimodal-rag-db"
GROQ_TEXT_MODEL = "openai/gpt-oss-20b"

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the documents provided. "
    "Use only the retrieved chunks. If they are not enough, say you don't know. "
    "Return a concise, clear answer. Use markdown when it improves readability. "
    "For math, use LaTeX: \\(inline\\) and \\[block\\]."
)

REFORMULATE_PROMPT = (
    "Given a chat history and the latest user question, which might reference "
    "context in the chat history, formulate a standalone question that can be "
    "understood without the chat history. Do not answer the question. "
    "If it is already standalone, return it unchanged. "
    "Return only the reformulated question."
)

chat_history = []


def load_vector_store() -> Chroma:
    """Open the existing multimodal Chroma database."""
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    return Chroma(
        persist_directory=MULTIMODAL_DB_DIR,
        embedding_function=embedding_model,
    )


def retrieve(db: Chroma, query: str, k: int = 3) -> list[Document]:
    """Return the top k chunks most similar to the query."""
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[Chunk {i}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_chat_history(history: list[dict]) -> str:
    lines = []
    for turn in history:
        role = "Human" if turn["role"] == "user" else "AI"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def reformulate_query(query: str, history: list[dict]) -> str:
    """Turn a follow-up into a standalone question using the chat history."""
    if not history:
        return query

    response = client.chat.completions.create(
        model=GROQ_TEXT_MODEL,
        messages=[
            {"role": "system", "content": REFORMULATE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Chat history:\n{format_chat_history(history)}\n\n"
                    f"Latest question: {query}"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or query).strip()


def generate_answer(query: str, docs: list[Document]) -> str:
    """Answer the question from the retrieved chunks using Groq."""
    user_message = (
        "Use only the following context to answer the question. "
        "If the context is not sufficient, say you don't know.\n\n"
        f"Context:\n{format_docs(docs)}\n\n"
        f"Question: {query}"
    )
    response = client.chat.completions.create(
        model=GROQ_TEXT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""


def main():
    db = load_vector_store()
    while True:
        query = input("Enter a question (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        standalone_query = reformulate_query(query, chat_history)
        print(f"Reformulated question: {standalone_query}")
        docs = retrieve(db, standalone_query, k=3)
        answer = generate_answer(standalone_query, docs)
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        print(answer)


if __name__ == "__main__":
    main()
