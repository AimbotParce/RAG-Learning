import argparse
import itertools
import json
import os
from collections import defaultdict
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List

import chromadb
import openai
import pandas as pd
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
    openai.api_key = os.getenv("EMBEDDING_KEY")
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
