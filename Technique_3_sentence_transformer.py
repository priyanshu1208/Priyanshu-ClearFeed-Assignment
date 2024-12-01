import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from sentence_transformers import SentenceTransformer
import faiss
import json
from openai_answer import query_openai, fetch_data_from_links

# Loading the documents
def load_documents_sent(documentation_json):
    document_texts = []
    document_urls = []

    for url, doc in documentation_json.items():
        document_texts.append(doc['text'])
        document_urls.append(url)

    return document_texts, document_urls

# Retrieving top 5 urls
def retrieve_top_5_urls_sent(query, documentation_json, embedding_model, document_embeddings=None, save_path='document_embeddings.npy'):
    if document_embeddings is None:
        document_embeddings = np.load(save_path)  
    document_texts, document_urls = load_documents_sent(documentation_json)
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)
    D, I = index.search(query_embedding.astype('float32'), k=5)  
    top_5_urls = [document_urls[i] for i in I[0]]
    return top_5_urls

# Fetching Answers using the links found above
def evaluate_sent(query, documentation_json):
    document_embeddings, _ = create_document_embeddings_sent(documentation_json)
    save_path='document_embeddings.npy'
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    top_5_urls = retrieve_top_5_urls_sent(query, documentation_json, embedding_model, document_embeddings, save_path)
    fetched_data = fetch_data_from_links(top_5_urls, documentation_json)
    openai_response = query_openai(query, fetched_data)
    return top_5_urls, openai_response

# Creating document embeddings using Sentence Transformer
def create_document_embeddings_sent(documentation_json, save_path='document_embeddings.npy'):

    document_texts = [doc['text'] for doc in documentation_json.values()]
    urls = list(documentation_json.keys())
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    document_embeddings = embedding_model.encode(document_texts, convert_to_numpy=True)
    
    np.save(save_path, document_embeddings)  
    
    return document_embeddings, urls




