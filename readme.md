# Semantic Search Assisted RAG

This project provides a simple semantic search system over a folder of documents using [ChromaDB](https://www.trychroma.com/) and [Sentence Transformers](https://www.sbert.net/). It includes both a script interface and a reusable Python API.

## Project Structure

- [`semanticSearch.py`](semanticSearch.py): Script to run semantic search on documents using a query from `query.txt`.
- [`semanticSearchApi.py`](semanticSearchApi.py): Python class for semantic search, suitable for integration into other projects.
- [`query.txt`](query.txt): Place your search query here (used by `semanticSearch.py`).
- `documents/`: Folder containing your text documents (one file per document).

## Requirements

- Python 3.8+
- [sentence-transformers](https://pypi.org/project/sentence-transformers/)
- [chromadb](https://pypi.org/project/chromadb/)

Install dependencies:

```sh
pip install sentence-transformers chromadb
```

## Usage

### 1. Prepare Documents

Place your `.txt` files inside the `documents/` folder. Each file will be treated as a separate document.

### 2. Run Semantic Search Script

1. Write your query in [`query.txt`](query.txt).
2. Run the script:

```sh
python semanticSearch.py
```

The script will print the top 3 most relevant documents and their similarity scores.

### 3. Use as a Python API

You can use the [`SemanticSearch`](semanticSearchApi.py) class in your own code:

```python
from semanticSearchApi import SemanticSearch

search = SemanticSearch(documents_folder="documents")
results = search.query("your search query here", n_results=3)
for result in results:
    print(result["name"], result["distance"])
    print(result["text"])
```

## How it Works

- Computes a hash of all files in `documents/` to detect changes.
- If documents change, rebuilds the ChromaDB vector store.
- Embeds documents and queries using `all-MiniLM-L6-v2` from Sentence Transformers.
- Performs semantic search to find the most relevant documents.

## License

MIT License