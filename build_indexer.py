#!/usr/bin/env python3
import os
import glob
from pathlib import Path
from typing import List
import re

import yaml
from chromadb import Settings
from multiprocessing import Pool
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

os.environ["NLTK_DATA"] = '/hps/nobackup/juan/pride/chatbot/'


class StructuredMarkdownLoader(UnstructuredFileLoader):
    
    def _get_elements(self) -> List:
        from unstructured.__version__ import __version__ as __unstructured_version__
        from unstructured.partition.md import partition_md

        # NOTE(MthwRobinson) - enables the loader to work when you're using pre-release
        # versions of unstructured like 0.4.17-dev1
        _unstructured_version = __unstructured_version__.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        if unstructured_version < (0, 4, 16):
            raise ValueError(
                f"You are on unstructured version {__unstructured_version__}. "
                "Partitioning markdown files is only supported in unstructured>=0.4.16."
            )
        filename = self.file_path 
        docs = []
        if filename.endswith(".md"):
            path = Path(filename)  # 替换为你的markdown文件路径
            content = path.read_text()
            sections = self.extract_sections(content)
            for section in sections:
                new_doc = Document(page_content=section.strip())
                docs.append(new_doc)
        return docs 
    
    @staticmethod
    def extract_sections(content: str) -> list:
        pattern = r"\n## |\n### |\n#### |\Z"
        sections = re.split(pattern, content)
        sections = [s.strip() for s in sections if s.strip()]
        return sections


# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        anonymized_telemetry=False, 
        persist_directory="",
)
os.environ["TOKENIZERS_PARALLELISM"] = "ture"  # Load the environment variables required by the local model


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".md": (StructuredMarkdownLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}

def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, doc in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.append(doc)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = [], source_directory: str = './documents', chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    return documents


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(
                os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def main(embeddings_model_name: str, persist_directory: str):
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        CHROMA_SETTINGS.persist_directory = persist_directory
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        CHROMA_SETTINGS.persist_directory = persist_directory
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory,
                                   client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run chatbotcli.py to query your documents")


if __name__ == "__main__":
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    embeddings_model_name = cfg['llm']['embedding']
    persist_directory = cfg['vector']['cli_store']+ cfg['vector']['uui'] + "/"
    main(embeddings_model_name, persist_directory) # call main function

