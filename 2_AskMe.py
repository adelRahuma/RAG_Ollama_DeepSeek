import os
import re
import faiss
import ollama
import pickle  # To load text chunks
from sentence_transformers import SentenceTransformer
 

FAISS_INDEX_PATH = "faiss_index"
TEXT_CHUNKS_PATH = "text_chunks.pkl"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

query = "how to install OpenVAS?"
query_embedding = embedding_model.encode([query]).astype('float32')

def load_text_chunks():
    if os.path.exists(TEXT_CHUNKS_PATH):
        with open(TEXT_CHUNKS_PATH, "rb") as f:
            return pickle.load(f)
    return []

index = faiss.read_index(FAISS_INDEX_PATH)
# Search FAISS index
k = 3  # Number of closest documents

distances, indices = index.search(query_embedding, k)
text_chunks = load_text_chunks()

if len(indices) == 0 or len(indices[0]) == 0:
    print("No results found in the FAISS index.")
else:
    retrieved_text = []
    for i in indices[0]:
        if 0 <= i < len(text_chunks):  # Ensure index is within bounds
            retrieved_text.append(text_chunks[i])
        else:
            print(f"Index {i} is out of bounds for text_chunks.")

    context = "\n".join(retrieved_text)

    # Step 4: Use Ollama to generate a response
    model_name = "deepseek-r1:1.5b"
    response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': f"Answer the question based on the following information:\n{context} only"}])

    # Clean the response (remove <think> tags)
    content = response['message']['content']
    cleaned_response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    print(cleaned_response)
