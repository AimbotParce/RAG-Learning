import argparse
import logging
import os
from typing import List, Literal, Optional, TypedDict

import chromadb
import openai
from dotenv import load_dotenv
from pydantic import BaseModel


class ChatEntry(TypedDict):
    role: Literal["developer", "user", "assistant"]
    content: str


def getEmbedding(prompt: str, embedding_model: str) -> List[float]:
    "Get the embedding of the prompt using the specified model"
    return openai.embeddings.create(input=prompt, model=embedding_model).data[0].embedding


class QueryDecision(BaseModel):
    "A decision to query the database"
    should_query: bool
    book_summary: Optional[str] = None


def decideQuery(user_prompt: str, query_decision_model: str) -> Optional[str]:
    "Decide whether or not to query the database based on a user's prompt"
    completion = openai.beta.chat.completions.parse(
        model=query_decision_model,
        messages=[
            {
                "role": "developer",
                "content": "Based on the following prompt, you decide whether or not to query a database of books."
                "If you decide to query the database, the book summary must be a rough invented summary of what the user is asking for."
                "In your query, don't include any names of books, characters, or authors except when the user prompt explicitly asks for them."
                "In your query, invent as little as possible, stick to the things that the user prompt is asking for."
                "You should only choose to query the database if the user prompt is asking for book recommendations, summaries or something like that.",
            },
            {"role": "user", "content": user_prompt},
        ],
        response_format=QueryDecision,
    )
    decision = completion.choices[0].message.parsed
    if decision.should_query:
        return decision.book_summary
    return None


def generateResponse(
    full_chat: List[ChatEntry], completion_model: str, context: Optional[List[dict[str, str]]] = None
):
    "Generate a response based on the full chat and the context"
    if context is not None:
        context_texts: list[str] = []
        for item in context:
            context_texts.append("\n".join([f"{key}: {value}" for key, value in item.items()]))
        context_text = "\n---\n".join(context_texts)
        context_text = "Here's information that might be relevant to answering the questions:\n" + context_text
    else:
        context_text = ""

    stream = openai.chat.completions.create(
        model=completion_model,
        messages=[
            {"role": "developer", "content": "You are an assistant." + context_text},
            *full_chat,
        ],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a RAG ChatGPT model")
    parser.add_argument(
        "--context-limit", type=int, default=3, help="The length of the context to retrieve from the database"
    )
    parser.add_argument("--history-limit", type=int, default=6, help="The length of the chat history to use")
    args = parser.parse_args()

    CONTEXT_LIMIT: int = args.context_limit
    HISTORY_LIMIT: int = args.history_limit

    # Load the environment variables
    load_dotenv()
    # Disable logging for chroma
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection("books")

    try:
        full_chat: list[ChatEntry] = []
        while True:
            prompt = input("(ctrl+C/bye to exit) > ")
            if prompt.lower() == "bye":
                print("Exiting...")
                break
            print()

            query = decideQuery(prompt, os.getenv("QUERY_DECISION_MODEL"))
            context = None
            if query:
                print("Retrieving some context...")
                embedding = getEmbedding(query, os.getenv("EMBEDDING_MODEL"))
                context = collection.query(query_embeddings=embedding, n_results=CONTEXT_LIMIT)["metadatas"][0]

            full_chat.append({"role": "user", "content": prompt})
            stream = generateResponse(full_chat[-HISTORY_LIMIT:], os.getenv("COMPLETION_MODEL"), context=context)
            response = ""
            for chunk in stream:
                response += chunk
                print(chunk, end="", flush=True)
            print("\n")
            full_chat.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("Exiting...")
