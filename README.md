# PDF Text Embedding with FAISS

This script extracts text from a PDF, chunks it, generates embeddings using `sentence-transformers`, and stores them in a FAISS index for efficient similarity search.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install faiss-cpu numpy sentence-transformers pymupdf

