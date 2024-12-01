<h1>ClearFeed Query Evaluation Techniques</h1>

Technique 1: BM25 (Okapi BM25)
BM25 is a probabilistic information retrieval model. It ranks a set of documents based on the query terms appearing in each document.

Technique 2: TF-IDF with Cosine Similarity
TF-IDF combined with cosine similarity measures the relevance of documents to the query based on term frequency.

Technique 3: Sentence Transformer with FAISS
This approach leverages Sentence Transformers for semantic embeddings and FAISS for efficient similarity search.

Technique 4: Google Search API Integration
This method queries Google Search for URLs relevant to the query within a specific domain.


<h2> Execution Steps</h2>
1. Install all the libraries from requirements.txt(pip install -r requirements.txt)
<br>
2. Add OPEN AI API key in .env file
<br>
3. Start the Flask server using app.py file(python app.py)
<br>

