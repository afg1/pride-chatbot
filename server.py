from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from peewee import fn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
import load_model
import os, re, gc, json, asyncio
import torch, threading
from fastapi import FastAPI, File, UploadFile, Request, Query, HTTPException, WebSocket, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import List
import json
import logging
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from queue import Queue
import threading
import time
import sqlite3
import markdown
from bs4 import BeautifulSoup

from chat_history import ChatHistory
from chat_history import ChatBenchmark

# global variables
os.environ["TOKENIZERS_PARALLELISM"] = "ture"  # Load the environment variables required by the local model
tokenizer = None
model = None
model_name_str = None
user_id = []
request_queue = Queue()
websockets = {}  # Saving websocket clients
UPLOAD_FOLDER = './documents/user_upload'


# functions

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


# Load the specified private database (vector) by specifying the id
def vector_by_id(path_id: str):
    directory = "./vector/" + path_id
    vector = Chroma(persist_directory=directory,embedding_function=HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2'))
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


# Search for relevant content in the vector based on the query and build a prompt
def get_similar_answer(vector, query, model) -> str:
    # prompt template, you can add external strings to { }
    if model == 'llama2-chat' or model == 'llama2-13b-chat':
        prompt_template = """
            <s>[INST]
            <<SYS>>
             You should summerize the knowledge and provide concise answer
            Please answer the questions according following Knowledge, and please convert the language of the generated answer to the same language as the user.
            If you does not know the answer to a question, please say I don’t know.
            ###Knowledge:{context}
            <</SYS>>
             ###Question:{question}
             [/INST]</s>
        """
    else:
        prompt_template = """
            You should summerize the knowledge and provide concise answer
            Please answer the questions according following Knowledge, and please convert the language of the generated answer to the same language as the user.
            If you does not know the answer to a question, please say I don’t know.
            ###Knowledge:{context}
            ###Question:{question}
        """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    docs = vector.similarity_search_with_score(query)
    print(docs)
    print('-------------------------------------------------------')
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
        document = document + str(count) + ':' + d[0].page_content + '\n [link](' + d[0].metadata['title'] + ')\n*******\n'
    # add the question input by user ande the relevant into prompt
    result = prompt.format(context='\t'.join(context), question=query)
    return result, document


# Processing chat requests
def process(prompt, model_name):
    torch.cuda.empty_cache()
    gc.collect()
    query = prompt
    db = Chroma(
        persist_directory="./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad370",
        embedding_function=HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2'))
    # Retrieve relevant documents in databse and form a prompt
    prompt, docs = get_similar_answer(vector=db, query=query, model=model_name)
    try:
        # tokenizer, model = load_model.llm_model_init(model_name, True)
        if model_name == 'llama2-13b-chat':
            completion = load_model.llm_chat(model_name, prompt, lltokenizer, llmodel, query)
        else:
            completion = load_model.llm_chat(model_name, prompt, glmtokenizer, glmmodel, query)
    except Exception as e:
        print(e)
        print('error in loading model', model_name)
        completion = "error in loading model"
    result = {"result": completion, "relevant-chunk": docs}
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

    result['timems']=time_ms

    return result


vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad370')

# interface

limiter = Limiter(key_func=get_remote_address)
# Create an app instance
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

lltokenizer, llmodel = load_model.llm_model_init('llama2-13b-chat', True)
glmtokenizer, glmmodel = load_model.llm_model_init('chatglm2-6b', True)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Accept foreground chat requests and place them in the queue
@app.post('/chat')
def chat(data: dict):
    return process_queue(data)

@app.get('/delete_all')
def chat():
    vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad370')
    vector.delete_collection()
    vector.persist()
    return {"status": "success"}

@app.post('/delete')
async def delete(item: dict):
    vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad370')
    filename = item["filename"]
    #os.remove(filename)
    print(filename)
    result = delete_by_file(vector, filename)
    return result

@app.post('/saveBenchmark')
def savebenchmark(data: dict):
    # insert the query & answer to database
    ChatBenchmark.create(query=data['query'], model_a=data['model_a'], model_b=data['model_b'],
                         answer_a=data['answer_a'], answer_b=data['answer_b'],
                         time_a=data['time_a'], time_b=data['time_b'],
                         winner=data['winner'], judge=data['judge'])


@app.get('/getBenchmark')
def getbenchmark(page_num: int = 0, items_per_page: int = 100):
    sql_results = ChatBenchmark.select(ChatBenchmark.query, ChatBenchmark.model_a, ChatBenchmark.model_b,
                                       ChatBenchmark.answer_a, ChatBenchmark.answer_b,
                                       ChatBenchmark.time_a, ChatBenchmark.time_b,
                                       ChatBenchmark.winner, ChatBenchmark.judge) \
        .paginate(page_num, items_per_page)

    results = []
    for row in sql_results:
        result_dict = {
            'query': row.query,
            'model_a': row.model_a,
            'answer_a': row.answer_a,
            'model_b': row.model_b,
            'answer_b': row.answer_b,
            'time_a': row.time_a,
            'time_b': row.time_b,
            'winner':row.winner,
            'judge': row.judge
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
async def upload(files: List[UploadFile] = File(...)):
    for file in files:
        if file.filename.endswith(".md"):
            docs = []
            content_bytes = await file.read()
            content = content_bytes.decode('utf-8')
            sections = extract_sections(content=content)
            parent_directory = os.path.dirname(file.filename)
            directory_name = os.path.basename(parent_directory)
            for section in sections:
                title = extract_title(content=section)
                html = markdown.markdown(section)
                soup = BeautifulSoup(html,'html.parser')
                new_doc = Document(
                    page_content=soup.get_text(),
                    metadata = {'source':UPLOAD_FOLDER+'/'+file.filename,
                                'title':"http://www.ebi.ac.uk/pride/markdownpage/"+directory_name+'#'+title,
                               })
                docs.append(new_doc)
            if len(docs)!=0:
                db = Chroma.from_documents(
                    documents=docs,
                    embedding=HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2'),
                    persist_directory="./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad370"
                )
                db.persist()
            directory = os.path.join(UPLOAD_FOLDER, os.path.dirname(file.filename))
            if not os.path.exists(directory):
                directory = os.path.join(UPLOAD_FOLDER, os.path.dirname(file.filename))
                os.makedirs(directory)
            with open(UPLOAD_FOLDER+'/'+file.filename, 'w', encoding='utf-8') as save_file:
                save_file.write(content)
    return json.dumps({'result':"update successful"})


# Download database
@app.get('/download')
def download_file(filename: str):
    return FileResponse(filename)


# Delete database
@app.post('/delete')
async def delete(item: dict):
    vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad370')
    filename = item["filename"]
    os.remove(filename)
    print(filename)
    result = delete_by_file(vector, filename)
    del vector
    return result

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
