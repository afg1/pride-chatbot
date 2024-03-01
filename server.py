import gc
import io
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from collections import defaultdict
from http.client import HTTPException
from queue import Queue
from urllib.parse import urlparse

import markdown
import torch
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, UploadFile
import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from peewee import fn, Query
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import load_model
from chat_history import ChatBenchmark, ProjectsSearchHistory
from chat_history import ChatHistory
from chat_history import QueryFeedBack

# global variables
os.environ["TOKENIZERS_PARALLELISM"] = "ture"  # Load the environment variables required by the local model
tokenizer = None
model = None
model_name_str = None
user_id = []
docs = []
docs_markdown = []
request_queue = Queue()
websockets = {}  # Saving websocket clients
UPLOAD_FOLDER = './documents/user_upload'


# functions

# functions
def create_visual(docs):
    first_children = []
    second_children = []
    for i in range(len(docs)):
        if i == 0:
            title = urlparse(docs[i].metadata['title']).fragment
            second_children.append({'name': title, "value": docs[i].metadata['title']})
        elif docs[i].metadata['source'] == docs[i - 1].metadata['source'] and i != len(docs) - 1:
            title = urlparse(docs[i].metadata['title']).fragment
            second_children.append({'name': title, "value": docs[i].metadata['title']})
        elif docs[i].metadata['source'] != docs[i - 1].metadata['source'] and i != len(docs) - 1:
            first_children.append(
                {"name": os.path.basename(os.path.dirname(docs[i - 1].metadata['source'])),
                 "value": docs[i - 1].metadata['title'].split('#')[0],
                 "children": second_children
                 })
            second_children = []
            title = urlparse(docs[i].metadata['title']).fragment
            second_children.append({'name': title, "value": docs[i].metadata['title']})
        elif i == len(docs) - 1:
            title = urlparse(docs[i].metadata['title']).fragment
            second_children.append({'name': title, "value": docs[i].metadata['title']})
            print(docs[i].metadata['source'])
            first_children.append(
                {"name": os.path.basename(os.path.dirname(docs[i].metadata['source'])),
                 "value": docs[i].metadata['title'].split('#')[0],
                 "children": second_children
                 })
    json_data = {
        "name": "markdown",
        "children": first_children
    }
    with open('./vector/tree.json', 'w') as file:
        json.dump(json_data, file)


# extrace ##title
def extract_title(content: str) -> str:
    titles = re.findall(r'^(#+)\s(.+)$', content, re.MULTILINE)
    for _, title in titles:
        title = title.lower()
        formatted_title = title.replace(" ", "_")
        formatted_title = formatted_title.replace(".", "")
        # formatted_title = formatted_title.replace(")", "\)")
        if formatted_title.endswith('_'):
            formatted_title = formatted_title[:-1]
    return formatted_title


# Split the content of the markdown file
def extract_sections(content: str) -> list:
    pattern = r"(\n# |\n## |\n### |\n#### |\Z)"
    sections = re.split(pattern, content)
    sections = [sections[i] + (sections[i + 1] if i + 1 < len(sections) else '') for i in range(1, len(sections), 2)]
    sections = [s.strip() for s in sections if s.strip()]
    return sections


# storage single file
def file_storage(file, content):
    global docs
    global docs_markdown
    parent_directory = os.path.dirname(file.filename)
    directory_name = os.path.basename(parent_directory).lower()
    sections = extract_sections("\n" + content)
    i = 0
    id_folder = str(uuid.uuid4())  # 识别同名文件所用到的id
    for section in sections:
        id = str(uuid.uuid4())
        new_doc_markdown = Document(
            page_content=section,
            metadata={'source': UPLOAD_FOLDER + '/' + id_folder + '/' + file.filename,
                      'id': id
                      })

        title = extract_title(content=section)
        html = markdown.markdown(section)
        soup = BeautifulSoup(html, 'html.parser')
        new_doc = Document(
            page_content=soup.get_text(),
            metadata={'source': UPLOAD_FOLDER + '/' + id_folder + '/' + file.filename,
                      'title': "http://www.ebi.ac.uk/pride/markdownpage/" + directory_name + '#' + title,
                      'id': id
                      })
        docs.append(new_doc)
        docs_markdown.append(new_doc_markdown)
    directory = os.path.join(UPLOAD_FOLDER, id_folder, os.path.dirname(file.filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(UPLOAD_FOLDER + '/' + id_folder + '/' + file.filename, 'w', encoding='utf-8') as save_file:
        save_file.write(content)
    return docs, docs_markdown


# Delete vector in chroma by filename
def delete_by_file(vector, filname: str):
    ids = []
    for i in range(len(vector.get()['metadatas'])):
        if filname == vector.get()['metadatas'][i]['source']:
            id = vector.get()['ids'][i]
            ids.append(id)
            print(id)
    if len(ids) != 0:
        vector.delete(ids)
        vector.persist()
        return json.dumps({"result": "success"})
    else:
        return json.dumps({"result": "Can't find the file"})


def find_same_markdown(vector, source) -> list:
    markdown_content = []
    for i in source:
        for j in range(len(vector.get()['metadatas'])):
            if i == vector.get()['metadatas'][j]['id']:
                markdown_content.append(vector.get()['documents'][j])
                break

    return markdown_content


def find_source(docs) -> list:
    source = []
    if len(docs) != 0:
        for d in docs:
            source.append(d[0].metadata['id'])
    else:
        source = None
    return source


# Load the specified private database (vector) by specifying the id
def vector_by_id(path_id: str):
    directory = "./vector/" + path_id
    vector = Chroma(persist_directory=directory,
                    embedding_function=HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2'))
    data = vector.get()['metadatas']
    unique_data = []
    seen = set()

    for item in data:
        identifier = item['source']
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(item)
    vector.source = unique_data
    return vector


def get_similar_projects_from_solr(accessions_str):
    accession_list = accessions_str.split(',')
    url = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{accession}/similarProjects?pageSize=10"
    accession_title_counts = defaultdict(int)

    for accession in accession_list:
        response = requests.get(url.format(accession=accession))
        if response.status_code == 200:
            data = response.json()
            compact_projects = data["_embedded"]["compactprojects"]
            for project in compact_projects:
                accession_title_counts[(project["accession"], project["title"])] += 1
        else:
            print(f"Failed to fetch data for accession {accession}")

    accession_title_list = [{"accession": accession, "title": title, "count": count} for (accession, title), count in
                            accession_title_counts.items()]
    sorted_accession_title_list = sorted(accession_title_list, key=lambda x: x["count"], reverse=True)
    return sorted_accession_title_list


# Search for relevant content in the vector based on the query and build a prompt
def get_similar_answer(vector, vector_markdown, query, model) -> str:
    # prompt template, you can add external strings to { }
    if model == 'llama2-chat' or model == 'llama2-13b-chat':
        prompt_template = """
            <s>[INST]
            <<SYS>>
             You should summerize the knowledge and provide concise answer
            Please answer the questions according following Knowledge, and please convert the language of the generated answer to the same language as the user.
            If you does not know the answer to a question, please say I don’t know.
            Knowledge:{context}
            <</SYS>>
             Question:{question}
             [/INST]</s>
        """
    elif model == 'Mixtral' or model == 'open-hermes':
        prompt_template = """[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
        Answer the question below from the context below:
        {context}
        {question} [/INST] 
        """
    else:
        prompt_template = """
            You are a helpful chatbot
            Please answer the questions according following Knowledge with markwown format
            Knowledge:{context}
            Question:{question}
        """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    docs = vector.similarity_search_with_score(query)
    if len(docs) != 0:
        source = find_source(docs)
        docs_markdown = find_same_markdown(vector_markdown, source)
        # put the relevant document into context
    document = ''
    count = 0
    context = []
    for d in docs:
        # if count==1:
        # context.append("Other relevant questions or info follows :")
        if count > 0:
            context.append("Other relevant questions or info follows :" + d[0].page_content)
        else:
            context.append(d[0].page_content)
        count += 1
        document = document + str(count) + ':' + docs_markdown[count - 1] + '\n [link](' + d[0].metadata[
            'title'] + ')\n*******\n'
    # add the question input by user ande the relevant into prompt
    result = prompt.format(context='\t'.join(context), question=query)
    return result, document


# Search for relevant content in the vector based on the query and build a prompt
def get_similar_answers_pride(vector, query, model) -> str:
    # prompt template, you can add external strings to { }
    if model == 'llama2-13b-chat':
        prompt_template = """
            <s>[INST]
            <<SYS>>
             You should summarize the knowledge and provide concise answer
            Please answer the questions according following Knowledge, and please convert the language of the generated answer to the same language as the user.
            If you does not know the answer to a question, please say I don’t know.
            Knowledge:{context}
            <</SYS>>
             Question:{question}
             [/INST]</s>
        """
    elif model == 'Mixtral' or model == 'open-hermes':
        prompt_template = """[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
        Answer the question below from the context below:
        {context}
        {question} [/INST] 
        """
    else:
        prompt_template = """
            You are a helpful chatbot
            Please answer the questions according following Knowledge with markdown format
            Knowledge:{context}
            Question:{question}
        """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    search_results = vector.similarity_search_with_score(query, k=10)
    count = 0
    accessions = []
    match_data = []
    for d in search_results:
        accessions.append(os.path.splitext(os.path.basename(d[0].metadata['source']))[0])
        match_data.append(d[0].page_content.split(".")[0])
        count += 1

    if count > 2:
        match_data.append("The above are top matching datasets , there could be others as well")
    if count == 0:
        match_data.append("No matching datasets found")

    accession_string = ' '.join(accessions)
    match_data_string = ' '.join(match_data)
    context = re.sub(r'[-:\n\s]+', ' ', match_data_string)
    ct = accession_string + " contains data " + context

    result = prompt.format(context=ct, question=query)
    accessions_delimited_by_comma = ','.join(accessions)
    return result, accessions_delimited_by_comma


# Processing chat requests
def process(prompt, model_name):
    torch.cuda.empty_cache()
    gc.collect()
    query = prompt
    # Retrieve relevant documents in databse and form a prompt
    result, docs = get_similar_answer(vector=db, vector_markdown=db_markdown, query=query, model=model_name)
    try:
        # tokenizer, model = load_model.llm_model_init(model_name, True)
        if model_name == 'llama2-13b-chat':
            completion = load_model.llm_chat(model_name, result, lltokenizer, llmodel, query)
        elif model_name == 'Mixtral':
            completion = load_model.llm_chat(model_name, result, mixtral_tokenizer, mixtral_model, query)
        elif model_name == 'open-hermes':
            completion = load_model.llm_chat(model_name, result, open_hermes_tokenizer, open_hermes_model, query)
        else:
            completion = load_model.llm_chat(model_name, result, glmtokenizer, glmmodel, query)
    except Exception as e:
        print(e)
        print('error in loading model', model_name)
        completion = "error in loading model"
    result = {"result": completion, "relevant-chunk": docs}
    return result


def process_pride_projects(prompt, model_name):
    torch.cuda.empty_cache()
    gc.collect()
    query = prompt
    # Retrieve relevant documents in database and form a prompt
    result, accessions = get_similar_answers_pride(vector=project_vector, query=query, model=model_name)
    try:
        # tokenizer, model = load_model.llm_model_init(model_name, True)
        if model_name == 'llama2-13b-chat':
            completion = load_model.llm_chat(model_name, result, lltokenizer, llmodel, query)
        elif model_name == 'Mixtral':
            completion = load_model.llm_chat(model_name, result, mixtral_tokenizer, mixtral_model, query)
        elif model_name == 'open-hermes':
            completion = load_model.llm_chat(model_name, result, open_hermes_tokenizer, open_hermes_model, query)
        else:
            completion = load_model.llm_chat(model_name, result, glmtokenizer, glmmodel, query)
    except Exception as e:
        print(e)
        print('error in loading model', model_name)
        completion = "error in loading model"
    result = {"result": completion, "relevant-chunk": accessions}
    return result


# Processing requests in the queue
def process_queue(data: dict):
    chat_query = data['prompt']
    chat_query = chat_query.strip()
    llm_model = data['model_name']

    start_time = round(time.time() * 1000)
    result = process(chat_query, llm_model)
    end_time = round(time.time() * 1000)

    result = {k: v.strip() for k, v in result.items()}
    time_ms = end_time - start_time

    # insert the query & answer to database
    ChatHistory.create(query=chat_query, model=llm_model, answer=result['result'], millisecs=time_ms)

    result['timems'] = time_ms

    return result


def process_pride(data: dict):
    chat_query = data['prompt']
    chat_query = chat_query.strip()
    llm_model = data['model_name']

    start_time = round(time.time() * 1000)
    result = process_pride_projects(chat_query, llm_model)
    end_time = round(time.time() * 1000)

    time_ms = end_time - start_time

    # insert the query & answer to database
    ProjectsSearchHistory.create(query=chat_query, model=llm_model, answer=result['result'], millisecs=time_ms)

    result['timems'] = time_ms

    return result


# interface

limiter = Limiter(key_func=get_remote_address)
# Create an app instance
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

lltokenizer, llmodel = load_model.llm_model_init('llama2-13b-chat', True)
glmtokenizer, glmmodel = load_model.llm_model_init('chatglm3-6b', True)
mixtral_tokenizer, mixtral_model = load_model.llm_model_init('Mixtral', True)
open_hermes_tokenizer, open_hermes_model = load_model.llm_model_init('open-hermes', True)

# Pride-docs vector database
db = vector_by_id("d4a1cccb-a9ae-43d1-8f1f-9919c90ad370")
db_markdown = vector_by_id("d4a1cccb-a9ae-43d1-8f1f-9919c90ad369")

# Pride-projects vector database
project_vector = vector_by_id("d4a1cccb-a9ae-43d1-8f1f-9919c90ad380")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

iterations = {
    1: "1stIteration.json",
    2: "2nd-iteration.json",
    3: "3rdIteration.json",
    4: "4thIteration.json",
}


@app.post('/chat')
def chat(data: dict):
    return process_queue(data)


@app.post('/pride')
def pride(data: dict):
    return process_pride(data)


@app.get('/similar_projects')
def pride(accessions: str):
    return get_similar_projects_from_solr(accessions)


@app.get('/delete_all')
def chat():
    vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad369')
    vector.delete_collection()
    vector.persist()
    vector = None
    vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad370')
    vector.delete_collection()
    vector.persist()
    vector = None
    if os.path.exists("./vector/tree.json") and os.path.exists("./documents/user_upload"):
        os.remove("./vector/tree.json")
        shutil.rmtree("./documents/user_upload")
    return {"status": "success"}


# Delete database
@app.post('/delete')
async def delete(item: dict):
    vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad370')
    vector_markdown = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad369')
    filename = item["filename"]
    os.remove(filename)
    print(filename)
    result = delete_by_file(vector, filename)
    result = delete_by_file(vector_markdown, filename)
    vector.persist()
    vector_markdown.persist()
    vector_markdown = None
    vector = None
    return result


@app.post('/saveBenchmark')
def savebenchmark(data: dict):
    # insert the query & answer to database
    ChatBenchmark.create(query=data['query'], model_a=data['model_a'], model_b=data['model_b'],
                         answer_a=data['answer_a'], answer_b=data['answer_b'],
                         time_a=data['time_a'], time_b=data['time_b'],
                         winner=data['winner'], judge=data['judge'])


@app.get('/getBenchmark')
def getbenchmark(page_num: int = 0, items_per_page: int = 100, iteration: int = 5):
    # Return the requested JSON file
    if iteration < 5:
        return FileResponse(iterations[iteration])
    elif iteration > 5:
        return '[]'

    sql_results = ChatBenchmark.select(ChatBenchmark.query, ChatBenchmark.model_a, ChatBenchmark.model_b,
                                       ChatBenchmark.answer_a, ChatBenchmark.answer_b,
                                       ChatBenchmark.time_a, ChatBenchmark.time_b,
                                       ChatBenchmark.winner, ChatBenchmark.judge) \
        .paginate(page_num, items_per_page)

    results = []
    for row in sql_results:
        if row.query != 'test':
            result_dict = {
                'query': row.query,
                'model_a': row.model_a,
                'answer_a': row.answer_a,
                'model_b': row.model_b,
                'answer_b': row.answer_b,
                'time_a': row.time_a,
                'time_b': row.time_b,
                'winner': row.winner,
                'judge': row.judge
            }
            results.append(result_dict)

    return results


@app.post('/saveQueryFeedBack')
def save_projects_query_feedback(data: dict):
    # insert the query & answer to database
    QueryFeedBack.create(query=data['query'],
                                 answer=data['answer'],
                                 feedback=data['feedback'])


@app.get('/getQueryFeedBack')
def get_projects_query_feedback(page_num: int = 0, items_per_page: int = 100):
    sql_results = QueryFeedBack.select(QueryFeedBack.query, QueryFeedBack.answer,
                                               QueryFeedBack.feedback) \
        .paginate(page_num, items_per_page)

    results = []
    for row in sql_results:
        result_dict = {
            'query': row.query,
            'answer': row.answer,
            'model': row.model,
            'time_ms': row.time_ms,
            'source': row.source,
            'feedback': row.feedback
        }
        results.append(result_dict)

    return results


# Load database
@app.get('/load')
async def load():
    # load the database according to uuid
    vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad370')
    return JSONResponse(content=vector.source)


# Update database
@app.post("/upload")
async def upload(files: UploadFile = File(...)):
    global docs
    docs_markdown = []
    if files.content_type == 'text/markdown':
        contents = await file.read()
        content = contents.decode("utf-8")
        file_storage(file, content)

    elif 'zip' in files.content_type:
        in_memory_file = io.BytesIO(await files.read())
        with zipfile.ZipFile(in_memory_file, 'r') as zip_ref:
            for file_info in zip_ref.infolist():

                print("storing file:" + file_info.filename)
                if not file_info.filename.startswith('__MACOSX'):
                    if file_info.filename.endswith('.md'):
                        print("storing file:" + file_info.filename)
                        with zip_ref.open(file_info) as md_file:
                            contents = md_file.read()
                            content = contents.decode("utf-8")
                            doc1, doc2 = file_storage(file_info, content)
                            gc.collect()
    # docs = sorted(docs, key=lambda doc: doc.metadata['source'])
    # docs_markdown = sorted(docs_markdown, key=lambda doc: doc.metadata['source'])
    create_visual(doc1)
    if len(doc1) != 0:
        db = Chroma.from_documents(
            documents=doc1,
            embedding=HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2"),
            persist_directory="./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad370"
        )
        db.persist()
        db = None
        db_markdown = Chroma.from_documents(
            documents=doc2,
            embedding=HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2"),
            persist_directory="./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad369"
        )
        db_markdown.persist()
        db_markdown = None
    return json.dumps({'result': "update successful"})


# Download database
@app.get('/download')
def download_file(filename: str):
    return FileResponse(filename)


@app.get('/get_tree')
def get_tree():
    with open('./vector/tree.json', 'r') as file:
        data = json.load(file)
    return json.dumps(data)


# Change model
@app.post('/model_choice')
async def model_choice(item: dict):
    global model_name_str
    model_name_str = item['model_name']
    return {"result": 'success'}


@app.post('/query_history')
def query_history(query: str, page_num: int = 0, items_per_page: int = 100):
    sql_results = ChatHistory.select(ChatHistory.query, ChatHistory.model, ChatHistory.answer,
                                     fn.AVG(ChatHistory.millisecs)) \
        .where(ChatHistory.query == query).group_by(ChatHistory.query, ChatHistory.model, ChatHistory.answer) \
        .paginate(page_num, items_per_page)

    results = []
    for row in sql_results:
        result_dict = {
            'query': row.query,
            'model': row.model,
            'answer': row.answer,
            'time': row.millisecs,
        }
        results.append(result_dict)

    return results


@app.get('/all_query_history')
def query_history(page_num: int = 0, items_per_page: int = 100):
    sql_results = ChatHistory.select(ChatHistory.query, ChatHistory.model, ChatHistory.answer,
                                     fn.AVG(ChatHistory.millisecs)) \
        .group_by(ChatHistory.query, ChatHistory.model, ChatHistory.answer).paginate(page_num, items_per_page)

    results = []
    for row in sql_results:
        result_dict = {
            'query': row.query,
            'model': row.model,
            'answer': row.answer,
            'time': row.millisecs,
        }
        results.append(result_dict)

    return results


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6008)
