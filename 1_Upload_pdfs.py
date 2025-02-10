import os
import faiss
import numpy as np
import pickle  # To save text chunks
from sentence_transformers import SentenceTransformer
import fitz  # pip install PyMuPDF


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")#384
FAISS_INDEX_PATH = "faiss_index"
TEXT_CHUNKS_PATH = "text_chunks.pkl"

# Step 1: Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
        # print(page.get_text("text"))
        # exit()
    return text

# Function to load existing text chunks
def load_text_chunks():
    if os.path.exists(TEXT_CHUNKS_PATH):
        with open(TEXT_CHUNKS_PATH, "rb") as f:
            return pickle.load(f)
    return []

# Function to save text chunks
def save_text_chunks(text_chunks):
    with open(TEXT_CHUNKS_PATH, "wb") as f:
        pickle.dump(text_chunks, f)

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])  # Extract chunk
        start = end - overlap  # Move start back by overlap to keep context
        # print(start)

    return chunks

# Function to add new text to the FAISS index
def append_to_faiss(pdf_path):
    # Load previous FAISS index if it exists, or create a new one
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        print("Loaded existing FAISS index.")
    else:
        index = faiss.IndexFlatL2(embedding_model.encode(["dummy"]).shape[1]) #KNN
        print("Created a new FAISS index.")

    # Load existing text chunks
    text_chunks = load_text_chunks()
    # Extract text from the new PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    new_text_chunks =chunk_text(pdf_text)

    # Append new chunks
    text_chunks.extend(new_text_chunks)
    # Convert text chunks to embeddings
    text_embeddings = embedding_model.encode(new_text_chunks)
    embedding_vectors = np.array(text_embeddings).astype('float32')
    index.add(embedding_vectors)
    faiss.write_index(index, FAISS_INDEX_PATH)
    save_text_chunks(text_chunks)
    print(f"FAISS index updated and saved to {FAISS_INDEX_PATH}!")
    return index, text_chunks  # Return updated index and text chunks


index, text_chunks = append_to_faiss("./uploads/VAS.pdf")  # Path to the new PDF file
print(" Document Uploaded Successfully!")