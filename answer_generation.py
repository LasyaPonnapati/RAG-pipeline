import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document

from retreival_pipeline import load_vector_store, retrieve

load_dotenv()

chat_history = []

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

REFORMULATE_PROMPT = (
    "Given a chat history and the latest user question, which might reference "
    "context in the chat history, formulate a standalone question that can be "
    "understood without the chat history. Do not answer the question. "
    "If it is already standalone, return it unchanged. "
    "Return only the reformulated question."
)

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

def format_chat_history(history: list[dict]) -> str:
    lines = []
    for turn in history:
        role = "Human" if turn["role"] == "user" else "AI"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)

def reformulate_query(query: str, history: list[dict]) -> str:
    if not history:
        return query

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
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
    return response.choices[0].message.content.strip()

def build_user_message(query: str, docs: list[Document]) -> str:
    context = format_docs(docs)
    return (
        "Use only the following context to answer the question. "
        "If the context is not sufficient, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

def generate_answer(query: str, docs: list[Document]) -> str:
    user_message = build_user_message(query, docs)
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content

def main():
    db = load_vector_store()
    while True:
        query = input("Enter a question(or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        standalone_query = reformulate_query(query, chat_history)
        print(f"Reformulated question: {standalone_query}")
        relevant_docs = retrieve(db, standalone_query)
        answer = generate_answer(standalone_query, relevant_docs)
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        print(answer)

if __name__ == "__main__":
    main()
