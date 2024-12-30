# RAG-Learning
A small scale project to learn about Retrieval Augmented Generation (RAG). This 
project provides a small chatbot that can answer book recommendation questions
using a database of books and their descriptions.

Prior to any response, the chatbot will determine whether the question requires
more context or if it can be answered directly. If more context is required, a
query will be generated and sent to the RAG model to retrieve the necessary
information. Finally, the chatbot will generate a response based on the retrieved
information.

## Scripts in this repository

These are the scripts that you need to know about in this repository:

- `insert_books.py`: This script reads the target .tsv file, with the same format
    as in [this kaggle dataset](https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset?resource=download),
    and inserts the books into the chroma database. Embeddings are generated from
    all the information in the book (title, author, genres, summary, etc.).
- `get.py`: This script retrieves a book's information from its title.
- `query.py`: Given a query, it will embed it and retrieve the ten most similar
    books to the query.
- `prompt.py`: Given a prompt, it will embed it, retrieve the five most similar
    books, and actually use them as context to generate a chatbot response.
- `chat.py`: The jewel in the crown. Run this script to start the chatbot. A
    supplementary chatbot will determine whether the user's messages require more
    context, and if so, they will be answered via Retrieval Augmentation.

