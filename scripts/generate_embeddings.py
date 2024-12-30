
from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np

DATA_FILE = "../data/courses.json"
INDEX_FILE = "../models/courses_index.faiss"

def generate_embeddings():
    # Load data
    with open(DATA_FILE, "r") as f:
        courses = json.load(f)

    # Initialize model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    descriptions = [course["description"] for course in courses]
    embeddings = model.encode(descriptions)

    # Create and save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_FILE)
    print(f"FAISS index saved to {INDEX_FILE}")

if __name__ == "__main__":
    generate_embeddings()
