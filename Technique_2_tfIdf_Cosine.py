import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
import openai
from openai_answer import query_openai, fetch_data_from_links


# Loading the documents
def load_documents_cosine(documentation_json):
    document_texts = []
    document_urls = []

    for url, doc in documentation_json.items():
        document_texts.append(doc['text'])
        document_urls.append(url)

    return document_texts, document_urls

# Document Embeddings using TfIdf
def create_document_embeddings_cosine(documentation_json, save_path='document_embeddings.npy'):
    document_texts, document_urls = load_documents_cosine(documentation_json)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(document_texts)
    np.save(save_path, tfidf_matrix.toarray()) 
    return tfidf_matrix.toarray(), document_urls, vectorizer

# Fetching top 5 urls
def retrieve_top_5_urls_cosine(query, vectorizer, document_embeddings, document_urls):
    query_vector = vectorizer.transform([query]).toarray()
    cosine_similarities = cosine_similarity(query_vector, document_embeddings).flatten()
    ranked_indices = np.argsort(cosine_similarities)[::-1][:5]
    top_5_urls = [document_urls[idx] for idx in ranked_indices]
    return top_5_urls

# Fetching Answers using the links found above
def evaluate_cosine(query, documentation_json):
    document_embeddings, document_urls, vectorizer = create_document_embeddings_cosine(documentation_json)
    top_5_urls = retrieve_top_5_urls_cosine(query, vectorizer, document_embeddings, document_urls)
    fetched_data = fetch_data_from_links(top_5_urls, documentation_json)
    openai_response = query_openai(query, fetched_data)
    

    return top_5_urls, openai_response

