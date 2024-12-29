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
from tqdm import tqdm

COLUMNS = ["WikipediaID", "FreebaseID", "BookTitle", "Author", "PublicationDate", "Genres", "Summary"]
CACHE_FOLDER = Path("cache")


def prepareDataset(books: pd.DataFrame) -> pd.DataFrame:
    "Prepare the dataset for insertion"

    # Drop the IDs, and the publication dates
    books = books.drop(columns=["WikipediaID", "FreebaseID", "PublicationDate"])

    def parseGenres(genres: str) -> str:
        "Parse the Genres. They are in the format of {'FreebaseID': 'Genre'}"
        if pd.isna(genres):
            return "Unknown"
        genres = json.loads(genres)
        return ", ".join(genres.values())

    books["Genres"] = books["Genres"].apply(parseGenres)
    return books


def constructPrompts(books: Iterable[pd.Series]) -> Iterator[str]:
    def constructPrompt(book: pd.Series) -> str:
        data: dict[str, str] = {}
        data["Book Title"] = book["BookTitle"] if not pd.isna(book["BookTitle"]) else "Unknown"
        data["Author"] = book["Author"] if not pd.isna(book["Author"]) else "Unknown"
        data["Genres"] = book["Genres"] if not pd.isna(book["Genres"]) else "Unknown"
        data["Summary"] = book["Summary"] if not pd.isna(book["Summary"]) else "Unknown"
        prompt = "\n".join([f"{key}: {value}" for key, value in data.items()])
        return prompt

    return map(constructPrompt, books)


def getEmbeddings(batched_prompts: Iterable[List[str]], embedding_model: str) -> Iterator[List[List[float]]]:
    def _getEmbeddings(texts: List[str]) -> List[List[float]]:
        "Get the embedding of the text using the specified model"
        if not isinstance(texts, (list, tuple)) or not all(isinstance(text, str) for text in texts):
            raise ValueError("texts should be a list of strings. Found: ", type(texts))
        if isinstance(texts, tuple):
            texts = list(texts)
        response_data = openai.embeddings.create(input=texts, model=embedding_model).data
        embeddings = list(map(lambda x: x.embedding, response_data))
        return embeddings

    return map(_getEmbeddings, batched_prompts)


class PipelineCache:
    """
    Cache the results of a pipelining function.
    For the sake of simplicity, this object only handles json-serializable data.
    """

    namespace: str = None  # Namespace for the cache. This must be set if cache_id is not provided

    _existing_instances: dict[str, int] = defaultdict(lambda: 0)

    def __init__(self, component: Callable[[Iterable, ...], Iterator], cache_id: str = None) -> None:
        if not isinstance(component, Callable):
            raise ValueError("component should be a callable")

        if cache_id is not None and not isinstance(cache_id, str):
            raise ValueError("id should be a string")

        if cache_id is None and self.namespace is None:
            raise ValueError("cache_id should be provided if namespace is not set")

        # If the cache_id is not provided, generate one from the namespace and
        # the component's name, bytecode, and a number that allows for multiple instances
        # of the same component to be cached
        if cache_id is None:
            code_hash = md5(component.__code__.co_code).hexdigest()
            cache_name = f"{self.namespace}_{component.__name__}_{code_hash}"
            cache_id = cache_name + f"_{self._existing_instances[cache_name]}"
            self._existing_instances[cache_name] += 1

        # If the cache folder doesn't exist, create it
        CACHE_FOLDER.mkdir(exist_ok=True)

        self._cache_id = cache_id
        self._component = component

    def __call__(self, input: Iterable, **kwargs: Any) -> Iterator:
        cache_file = CACHE_FOLDER / f"{self._cache_id}.cache"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                for line in f:
                    next(input)  # Skip the first n elements
                    yield json.loads(line)  # Yield the cached data
        with open(cache_file, "a") as f:
            for data in self._component(input, **kwargs):
                f.write(json.dumps(data) + "\n")
                yield data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert books in the database")
    parser.add_argument("books", type=Path, help="Path to the tsv file with the books")
    args = parser.parse_args()

    # Load the environment variables
    load_dotenv()
    openai.api_key = os.getenv("EMBEDDING_KEY")
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection("books")
    PipelineCache.namespace = args.books.stem

    books = pd.read_csv(args.books, sep="\t", names=COLUMNS)
    books = prepareDataset(books)

    # We'll pipeline the process, and cache the steps so we don't have to recompute them
    books_iterator = map(lambda b: b[1], books.iterrows())
    prompts = PipelineCache(constructPrompts)(books_iterator)

    cut_prompts = map(lambda x: x[:8192], prompts)  # Trim the prompts (embeddings has a maximum of 8192 tokens)
    batched_prompts = itertools.batched(cut_prompts, 64)  # Batch the prompts in groups of 64
    embeddings = PipelineCache(getEmbeddings)(batched_prompts, embedding_model=os.getenv("EMBEDDING_MODEL"))
    embeddings = itertools.chain.from_iterable(embeddings)  # Flatten the embeddings

    for (j, book), embedding in tqdm(zip(books.iterrows(), embeddings), total=len(books), leave=False):
        book: dict[str, str] = book.to_dict()
        title = book["BookTitle"]
        author = book["Author"] if not pd.isna(book["Author"]) else "Unknown"
        book_id = f"{title} - {author}"
        collection.add(ids=book_id, embeddings=embedding, metadatas=book, uris=title)
