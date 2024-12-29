import argparse

import chromadb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get all information about a book")
    parser.add_argument("title", type=str, help="The book title")
    args = parser.parse_args()

    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection("books")

    # Query the database
    result = collection.get(where={"BookTitle": args.title})
    metadatas = result["metadatas"]
    if not metadatas:
        print("Book not found")
        exit(1)
    print(f"Found {len(metadatas)} book(s):")
    for result in metadatas:
        print("-" * 30)
        print("Title:", result["BookTitle"])
        print("Author:", result.get("Author", "Unknown"))
        print("Genres:", result.get("Genres", "Unknown"))
        print("Summary:", result.get("Summary", "Unknown"))
