import os
import re
import uuid

import markdown
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

UPLOAD_FOLDER = './documents/user_upload'


# create vector database from one folder directory
def create_and_save(file_path: str):
    file_id = str(uuid.uuid4())
    save_path = "./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad380"
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    docs, docs_markdown = import_file(file_path)
    vector = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=save_path
    )
    vector.id = "d4a1cccb-a9ae-43d1-8f1f-9919c90ad380"

    save_path1 = "./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad379"
    vector1 = Chroma.from_documents(
        documents=docs_markdown,
        embedding=embeddings,
        persist_directory=save_path1
    )
    vector1.id = "d4a1cccb-a9ae-43d1-8f1f-9919c90ad379"
    return vector, vector1


# import all markdown files in the folder and then do the segmentation
def import_file(data_folder: str) -> list:
    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            fullPath = os.path.join(root, filename)
            if filename.endswith(".md") and filename.startswith("PXD"):
                docs = []
                docs_markdown = []
                content = open(fullPath, 'r', encoding='utf-8').read()
                sections = extract_sections(content=content)
                parent_directory = os.path.dirname(fullPath)
                directory_name = os.path.basename(parent_directory)
                for section in sections:
                    html = markdown.markdown(section)
                    soup = BeautifulSoup(html, 'html.parser')
                    meta_id = str(uuid.uuid4())
                    new_doc = Document(
                        page_content=soup.get_text(),
                        metadata={'source': UPLOAD_FOLDER + '/' + directory_name + '/' + filename,
                                  'title': "http://www.ebi.ac.uk/pride/" + filename.split('.')[0],
                                  'id': meta_id
                                  })
                    docs.append(new_doc)
                    new_doc_markdown = Document(
                        page_content=section,
                        metadata={'source': UPLOAD_FOLDER + '/' + directory_name + '/' + filename,
                                  'id': meta_id
                                  })
                    docs_markdown.append(new_doc_markdown)
                if len(docs) != 0:
                    db = Chroma.from_documents(
                        documents=docs,
                        embedding=HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2'),
                        persist_directory="./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad380"
                    )
                    db.persist()
                directory = os.path.join(UPLOAD_FOLDER, directory_name)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                with open(directory + '/' + filename, 'w', encoding='utf-8') as save_file:
                    save_file.write(content)
    return docs, docs_markdown


# extrace ##title
def extract_title(content: str) -> str:
    titles = re.findall(r'^(#+)\s(.+)$', content, re.MULTILINE)
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


# main function
if __name__ == '__main__':
    # start service
    create_and_save('./data')
