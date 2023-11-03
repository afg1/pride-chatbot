import requests
import json
import os
import uuid
import re
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from pathlib import Path
import markdown
from bs4 import BeautifulSoup

# URL to fetch JSON data from
url = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/metadata"

try:
    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON data from the response
        data = response.json()

        db = Chroma.from_documents(
            documents=data,
            embedding=HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2'),
            persist_directory="./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad380"
        )
        db.persist()

        print(f"JSON data has been successfully saved to {filename}")
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
except json.JSONDecodeError as e:
    print(f"Failed to decode JSON data: {e}")