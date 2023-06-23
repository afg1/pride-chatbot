import os 
import uuid
import re
from langchain.vectorstores import Chroma 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from pathlib import Path

# create vector database from one folder directory
def create_and_save(file_path:str):
    file_id = str(uuid.uuid4())
    save_path = "./vector_store/"+file_id
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = import_file(file_path)
    Vector= Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=save_path
    )
    Vector.persist()
    Vector.id = file_id
    return Vector

# import all markdown files in the folder and then do the segmentation
def import_file(data_folder: str) -> list:
    docs = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            filename = os.path.join(root, file)
            if filename.endswith(".md"):
                path = Path(filename)  # 替换为你的markdown文件路径
                content = path.read_text()
                sections = extract_sections(content)
                for section in sections:
                    new_doc = Document(page_content=section.strip())
                    docs.append(new_doc)
    for doc in docs: 
        print(doc)
    return docs

# used by the above import_file function to do the segmentation. 
# This function plays the role like "RecursiveCharacterTextSplitter" from LangChain.
# But it is not good to use "RecursiveCharacterTextSplitter" to do the segmentation only according to "chunk_size".
# I have explain the reason in "build_indexer.py" file in the function "process_documents"
def extract_sections(content: str) -> list:
    pattern = r"\n## |\n### |\n#### |\Z"
    sections = re.split(pattern, content)
    sections = [s.strip() for s in sections if s.strip()]
    return sections

#update vector databse via REST API used in server.py, we could ignore it currently as we only use command line to do it.  
def add_with_id(data: str, path_id: str):
    save_path = "./vector_store/" + path_id
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    doc = Document(page_content = data,metadata={"page": "0"})
    Vector = Chroma.from_documents(
        documents = [doc],
        embedding=embeddings,
        persist_directory=save_path
    )
    Vector.id = path_id
    return Vector


#main function
if __name__ == '__main__':
    
    #start service
    src= input('Please input absolute path of all markdown files')
    create_and_save(src)


