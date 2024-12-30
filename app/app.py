import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the scraped data
DATA_FILE = "data/courses.json"

def load_data():
    """
    Load course data from the JSON file.
    """
    with open(DATA_FILE, "r") as f:
        return json.load(f)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

@st.cache_resource
def generate_embedding(text):
    """
    Generate embeddings for the input text using BERT.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def search_courses(query, data):
    """
    Perform course search based on query similarity to course titles/descriptions.
    """
    query_embedding = generate_embedding(query)

    # Create a 2D array for course embeddings
    course_embeddings = np.array([
        generate_embedding(course["title"] + " " + course.get("description", ""))
        for course in data
    ])

    # Ensure all embeddings are 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, course_embeddings).flatten()

    # Assign similarity scores to courses
    for i, course in enumerate(data):
        course["score"] = similarities[i]

    # Return top 10 results sorted by similarity score
    return sorted(data, key=lambda x: x["score"], reverse=True)[:10]

# Streamlit UI
def main():
    """
    Main function to render the Streamlit app.
    """
    # Add custom CSS for styling
    st.markdown("""
        <style>
        body {
            background-color: #1e1e2f;
            color: #ffffff;
        }
        .stApp {
            background-color: #1e1e2f;   
        }
        .title {
            text-align: center;
            font-size: 3rem;
            color: #42a5f5;
            text-shadow: 2px 2px 5px #000000;
        }
        .subtitle {
            text-align: center;
            font-size: 1.5rem;
            color: #90caf9;
            text-shadow: 2px 2px 10px yellow;
        }
        .stColumns {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: auto;
            max-width: 90%; /* Adjusted for a wider layout */
        }
        .card {
            background-color: #2b2b3d;
            padding: 14px;
            margin: 12px 2px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5);
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
        }
        .card:hover {
            transform: scale(1.05);
            box-shadow: 2px 2px 20px rgba(0, 0, 0, 0.8);
        }
        .card img {
            width: 100%;
            border-radius: 10px;
            height: 200px;
            object-fit: cover;
        }
        .card-title a {
            text-align: left;
            font-size: 1.2rem;
            color: #64b5f6;
            margin: 0px;
            text-decoration: none;
            font-weight: bold;
        }
        .card-title a:hover {
            color: #2196f3;
            text-decoration: underline;
        }
        .card-desc {
            color: #b0bec5;
            margin: 10px 20px;
        }
        .score {
            font-size: 1.1rem;
            color: #ffeb3b;
            font-weight: bold;
            margin-top: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Page Title
    st.markdown('<h1 class="title">Analytics Vidhya Smart Search</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle">Find free courses relevant to your interests.</h3>', unsafe_allow_html=True)

    # Search Input
    query = st.text_input("Search for a course", placeholder="e.g., Machine Learning, Python")
    if query:
        data = load_data()
        results = search_courses(query, data)

        if results:
            st.markdown(f"### Showing results for: {query}")

            # Dynamically create rows with cards
            for i in range(0, len(results), 2):  # 2 cards per row
                cols = st.columns(2)  # Create two columns
                for idx, result in enumerate(results[i:i + 2]):
                    with cols[idx]:
                        st.markdown(f"""
                            <div class="card">
                                <img src="{result['image_url']}" alt="Course Image">
                                <h2 class="card-title">
                                    <a href="{result['course_link']}" target="_blank">{result['title']}</a>
                                </h2>
                                <p class="card-desc">{result.get('description', 'No description available.')}</p>
                                <p class="score">Relevance Score: {round(result['score'] * 100, 2)}%</p>
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.write("No courses found. Please try a different query.")

if __name__ == "__main__":
    main()
