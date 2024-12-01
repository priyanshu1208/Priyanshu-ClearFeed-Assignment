import pandas as pd
import numpy as np
import random
import requests
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import json
import nltk
import re
import openai
from openai_answer import query_openai, fetch_data_from_links


# Loading the document
def load_documents_bm25(documentation_json):
    document_texts = []
    document_urls = []

    for url, doc in documentation_json.items():
        document_texts.append(doc['text'])
        document_urls.append(url)

    return document_texts, document_urls


# Retrieving top 5 urls
def retrieve_top_5_bm25(query, document_texts, document_urls):
    tokenized_docs = []
    for doc in document_texts:
        tokenized_docs.append(word_tokenize(doc.lower()))

    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = np.argsort(scores)[::-1]
    top_5_urls = []
    for i in ranked_indices[:5]:
        top_5_urls.append(document_urls[i])

    return top_5_urls


# Fetching Answers using the links found above
def evaluate_bm25(query, documentation_json):
    print("BM25")
    
    document_texts, document_urls = load_documents_bm25(documentation_json)
    top_5_urls = retrieve_top_5_bm25(query, document_texts, document_urls)
    
    fetched_data = fetch_data_from_links(top_5_urls, documentation_json)
    openai_response = query_openai(query, fetched_data)
    
    return top_5_urls, openai_response

