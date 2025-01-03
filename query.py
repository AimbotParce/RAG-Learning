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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the database")
    parser.add_argument("query", type=str, help="The query to run")
    args = parser.parse_args()

    # Load the environment variables
    load_dotenv()
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection("books")

    # Get the embedding of the query
    embedding = getEmbedding(args.query, os.getenv("EMBEDDING_MODEL"))

    # Query the database
    results = collection.query(query_embeddings=embedding)
    for result in results["metadatas"][0]:
        print("Title:", result["BookTitle"])
        print("Author:", result.get("Author", "Unknown"))
        print("Genres:", result.get("Genres", "Unknown"))
        print("-" * 30)

    print("If you want to see more details, call `get.py <book_title>`")
