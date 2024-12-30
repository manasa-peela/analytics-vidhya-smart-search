import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Path to the scraped data
DATA_FILE = "data/courses.json"

# Load pre-trained BERT model and tokenizer
@torch.no_grad()
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_bert_model()

# Load scraped course data
def load_course_data(file_path):
    """
    Load course data from JSON file.
    """
    with open(file_path, "r") as f:
        return json.load(f)

courses = load_course_data(DATA_FILE)

# Function to generate embeddings using BERT
def get_bert_embedding(text):
    """
    Generate BERT embedding for the input text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Pre-compute embeddings for all courses
def precompute_embeddings(courses):
    """
    Precompute BERT embeddings for course titles and descriptions.
    """
    for course in courses:
        course["embedding"] = get_bert_embedding(
            course["title"] + " " + course.get("description", "")
        )
    return courses

courses = precompute_embeddings(courses)

# Function to search courses based on query
def search_courses(query, top_n=5):
    """
    Search courses based on a query using cosine similarity.
    """
    query_embedding = get_bert_embedding(query)
    course_embeddings = np.vstack([course["embedding"] for course in courses])

    # Compute cosine similarity between query and course embeddings
    similarities = cosine_similarity(query_embedding, course_embeddings).flatten()

    # Add similarity scores to courses
    for i, course in enumerate(courses):
        course["score"] = similarities[i]

    # Sort courses by similarity score
    sorted_courses = sorted(courses, key=lambda x: x["score"], reverse=True)
    return sorted_courses[:top_n]

# Test the search system
if __name__ == "__main__":
    print("Welcome to the Course Search Engine!")
    while True:
        query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting the search engine. Goodbye!")
            break

        results = search_courses(query)
        
        if results:
            print("\nTop Results:\n")
            for idx, result in enumerate(results, 1):
                print(f"{idx}. Title: {result['title']}")
                print(f"   Description: {result.get('description', 'No description available.')}")
                print(f"   Link: {result['course_link']}")
                print(f"   Relevance: {round(result['score'] * 100, 2)}%")
                print("-" * 50)
        else:
            print("No matching courses found. Please try another query.")
