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

UPLOAD_FOLDER = './documents/user_upload'

# create vector database from one folder directory
def create_and_save(file_path:str):
    file_id = str(uuid.uuid4())
    save_path = "./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad370"
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    docs = import_file(file_path)
    Vector= Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=save_path
    )
    Vector.id = "d4a1cccb-a9ae-43d1-8f1f-9919c90ad370"
    return Vector

# import all markdown files in the folder and then do the segmentation
def import_file(data_folder: str) -> list:
    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            fullPath = os.path.join(root, filename)
            if filename.endswith(".md"):
                docs = []
                print(filename)
                content = open(fullPath, 'r', encoding='utf-8').read()
                sections = extract_sections(content=content)
                parent_directory = os.path.dirname(fullPath)
                directory_name = os.path.basename(parent_directory)
                for section in sections:
                    title = extract_title(content=section)
                    html = markdown.markdown(section)
                    soup = BeautifulSoup(html, 'html.parser')
                    new_doc = Document(
                        page_content=soup.get_text(),
                        metadata={'source': UPLOAD_FOLDER + '/' + directory_name + '/' +filename,
                                  'title': "http://www.ebi.ac.uk/pride/markdownpage/" + directory_name + '#' + title,
                                  })
                    docs.append(new_doc)
                if len(docs) != 0:
                    db = Chroma.from_documents(
                        documents=docs,
                        embedding=HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2'),
                        persist_directory="./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad370"
                    )
                    db.persist()
                directory = os.path.join(UPLOAD_FOLDER, directory_name)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                with open(directory + '/' + filename, 'w', encoding='utf-8') as save_file:
                    save_file.write(content)
    return docs
#extrace ##title
def extract_title(content:str)-> str:
    titles = re.findall(r'^(#+)\s(.+)$',content, re.MULTILINE)
    for _, title in titles:
        title = title.lower()
        formatted_title = title.replace(" ", "_")
    return formatted_title

# Split the content of the markdown file
def extract_sections(content: str) -> list:
    pattern = r"(\n# |\n## |\n### |\n#### |\Z)"
    sections = re.split(pattern, content)
    sections = [sections[i] + (sections[i + 1] if i + 1 < len(sections) else '') for i in range(1, len(sections), 2)]
    sections = [s.strip() for s in sections if s.strip()]
    return sections


#main function
if __name__ == '__main__':

    #start service
    create_and_save('./documents/pride/')
