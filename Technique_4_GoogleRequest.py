import requests
import urllib.parse
from bs4 import BeautifulSoup
from openai_answer import fetch_data_from_links,query_openai
import json

def google_search(query, site):
    base_url = "https://www.google.com/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    }
    query_encoded = urllib.parse.quote_plus(query)
    params = {
        "q": f"site: {site} {query_encoded}"
    }
    final_query_url = f"{base_url}?q=site:{site}+{query_encoded}"
    print(f"Final Google Search URL: {final_query_url}")

    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Error: Unable to fetch results, Status Code: {response.status_code}")

#  Extracting top matching urls
def extract_desired_urls(html_content, desired_base_url):
    soup = BeautifulSoup(html_content, "html.parser")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith(desired_base_url):
            links.append(href)
    return links

def main(query,documentation_json):
    site = "docs.clearfeed.ai"
    desired_base_url = "https://docs.clearfeed.ai/"
    try:
        html_content = google_search(query, site)
        urls = extract_desired_urls(html_content, desired_base_url)
        fetched_data = fetch_data_from_links(urls, documentation_json)
        openai_response = query_openai(query, fetched_data)

        return urls,openai_response
    except Exception as e:
        print(f"An error occurred: {e}")