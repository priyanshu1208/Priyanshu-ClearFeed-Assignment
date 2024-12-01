
import os
from dotenv import load_dotenv
import openai


load_dotenv()


openai_api_key = os.getenv('OPENAI_API_KEY')

def query_openai(query, fetched_data):
    formatted_data = "\n".join([f"Link: {item['link']}\nContent: {item['text']}" for item in fetched_data])
    system_message = "You are a helpful assistant. Use only the provided data to answer queries in 200 words in 2 paragraphs."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Data:\n{formatted_data}"},
        {"role": "user", "content": f"Query: {query}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )
    
    return response.choices[0].message.content

def fetch_data_from_links(links, documentation_json):
    content = []
    for link in links:
        if link in documentation_json:
            content.append({"link": link, "text": documentation_json[link]['text']})
        else:
            content.append({"link": link, "text": "Content not found in the provided JSON."})

    return content
