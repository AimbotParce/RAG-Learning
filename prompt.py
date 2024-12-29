import argparse
import os
from typing import List

import chromadb
import openai
from dotenv import load_dotenv


def getEmbedding(prompt: str, embedding_model: str) -> List[float]:
    "Get the embedding of the prompt using the specified model"
    if not isinstance(prompt, str):
        raise ValueError("Prompt should be a string. Found: ", type(prompt))
    return openai.embeddings.create(input=prompt, model=embedding_model).data[0].embedding


def getResponse(prompt: str, context: list[dict[str, str]], completion_model: str) -> str:
    "Respond to the prompt using the context and the completion model"
    if not isinstance(prompt, str):
        raise ValueError("Prompt should be a string. Found: ", type(prompt))
    if not isinstance(context, list):
        raise ValueError("Context should be a list. Found: ", type(context))

    context_texts: list[str] = []
    for item in context:
        context_texts.append("\n".join([f"{key}: {value}" for key, value in item.items()]))
    context_text = "\n---\n".join(context_texts)

    completion = openai.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": "You are a book recommender. You answer questions about books, or recommend books."
                "Here's information about some books that might be relevant to answering the questions:\n"
                + context_text,
            },
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt the database")
    parser.add_argument("prompt", type=str, help="A prompt for ChatGPT and the database")
    args = parser.parse_args()

    # Load the environment variables
    load_dotenv()
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection("books")

    # Get the embedding of the query
    print("Analyzing your prompt...")
    embedding = getEmbedding(args.prompt, os.getenv("EMBEDDING_MODEL"))

    print("Retrieving some context...")
    context = collection.query(query_embeddings=embedding, n_results=5)["metadatas"][0]

    print("Generating a response...")
    response = getResponse(args.prompt, context, os.getenv("COMPLETION_MODEL"))
    print("")
    print(response)
