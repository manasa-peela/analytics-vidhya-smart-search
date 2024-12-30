
# Analytics Vidhya Smart Search

## Overview
This project creates a semantic search tool for free courses available on the Analytics Vidhya platform. Users can search using keywords or natural language queries.

## Setup
1. Install dependencies using `pip install -r requirements.txt`.
2. Run the Streamlit app using `streamlit run app/app.py`.

## Technical Choices
- **Embedding Model**: `all-MiniLM-L6-v2` from sentence-transformers.
- **Vector Database**: FAISS for efficient similarity search.
- **Frontend Framework**: Streamlit for a user-friendly interface.

## How It Works
1. Scrapes course details (titles, images, and links) from the Analytics Vidhya website.
2. Uses BERT embeddings to calculate the semantic similarity between user queries and course titles.
3. Displays results in a user-friendly Streamlit interface.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repo_url>
   cd <repo_directory>
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the scraper:
   ```
   python scripts/scrape_data.py
   ```
4. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

## Example Query
Search for terms like "Python," "Machine Learning," or "AI" to find relevant courses.
```

---

## How It's Different
1. **Original Scraping Logic**:
   - The scraping logic uses a slightly different approach for extracting course details.

2. **Independent Implementation**:
   - The logic for embedding generation and cosine similarity is written from scratch.

3. **Clean Structure**:
   - The project is well-organized with clear distinctions between scripts for scraping, searching, and UI.

