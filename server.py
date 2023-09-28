from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel  
from langchain.docstore.document import Document  
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
import load_model
import os,re,gc,json,asyncio
import torch,threading
from fastapi import FastAPI, File, UploadFile, Request, Query,HTTPException,WebSocket,BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import List
import json
import logging
from typing import Dict,Any
from fastapi.middleware.cors import CORSMiddleware
from queue import Queue
import threading
import time
import sqlite3

# global variables
os.environ["TOKENIZERS_PARALLELISM"] = "ture"  #Load the environment variables required by the local model
tokenizer  = None
model = None
model_name_str = None
user_id = []
request_queue = Queue()
websockets = {}  # Saving websocket clients
UPLOAD_FOLDER = './documents/user_upload'
sql_conn = sqlite3.connect('chatbot.db')

# functions

#Split the content of the markdown file
def extract_sections(content: str) -> list:
    pattern = r"\n## |\n### |\n#### |\Z"
    sections = re.split(pattern, content)
    sections = [s.strip() for s in sections if s.strip()]
    return sections


#Delete vector in chroma by filename
def delete_by_file(vector,filname:str):
    ids = []
    for i in range(len(vector.get()['metadatas'])):
        if filname == vector.get()['metadatas'][i]['source']:
            id = vector.get()['ids'][i]
            ids.append(id)
            print(id)
    if len(ids)!=0:
        vector.delete(ids)
        vector.persist()
        return jsonify({"result":"success"})
    else:
        return jsonify({"result":"Can't find the file"})


#Load the specified private database (vector) by specifying the id
def vector_by_id(path_id:str):
        directory = "./vector/"+path_id
        vector = Chroma(persist_directory=directory, embedding_function=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))
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


#Search for relevant content in the vector based on the query and build a prompt
def get_similar_answer(vector, query,model) -> str:
    #prompt template, you can add external strings to { }
    if  model == 'llama2-chat':
        
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
    #put the relevant document into context
    document = ''
    count = 0
    context = []
    for d in docs:
        if d[1] < 1:
            context.append(d[0].page_content)
            count+=1
            document = document+str(count)+':'+d[0].page_content+'\n*******\n'
    #add the question input by user ande the relevant into prompt
    result = prompt.format(context='\t'.join(context), question=query)
    return result,document


#Processing chat requests
def process(prompt, model_name):
    global model
    global tokenizer
    global model_name_str 
                
    if model_name_str!= model_name:
        del model
        del tokenizer
        model_name_str= model_name
        torch.cuda.empty_cache()
        gc.collect()
        tokenizer,model =load_model.llm_model_init(model_name_str,True)
    query = prompt
    db = Chroma(persist_directory="./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad369", embedding_function=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))
    #Retrieve relevant documents in databse and form a prompt
    prompt,docs = get_similar_answer(vector = db,query = query,model =  model_name)
    completion = load_model.llm_chat(model_name,prompt,tokenizer,model,query)
    result = {"result": completion,"relevant-chunk": docs}
    return result     


#Processing requests in the queue
def process_queue():
    while True:
        if not request_queue.empty():
            data =  request_queue.get()
            start_time = round(time.time() * 1000)
            chat_query = data['prompt']
            chat_query = chat_query.strip()
            llm_model = data['model_name']
            result = process(chat_query, llm_model)
            result = result.strip()
            end_time = round(time.time() * 1000)
            time_ms = end_time - start_time

            # insert the query & answer to database
            global sql_conn
            cursor = sql_conn.cursor()
            cursor.execute("INSERT INTO chat_history (query, model, answer, millisecs) VALUES (?, ?, ?, ?)", (chat_query, llm_model, result, time_ms))
            sql_conn.commit()
            cursor.close()

            data['response_queue'].put( result )
            request_queue.task_done()


def create_sqlite_db():
    global sql_conn
    sql_cursor = sql_conn.cursor()

    sql_cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY,
            query TEXT,
            model TEXT,
            answer TEXT,
            millisecs INTEGER
        )
    ''')

    sql_cursor.close()
    # sql_conn.close()


create_sqlite_db()

processing_thread = threading.Thread(target= process_queue)
processing_thread.daemon = True
processing_thread.start()


# interface

#Create an app instance 
app = FastAPI()

#CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Accept foreground chat requests and place them in the queue
@app.post('/chat')
async def chat(data: dict):
    global model_name_str 
    global user_id
    response_queue = Queue() 
    data['response_queue'] = response_queue
    request_queue.put(data)
    # 等待处理结果
    result = response_queue.get()
    return result

#Load database
@app.get('/load')
async def load():
    #load the database according to uuid
    vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad369')
    return JSONResponse(content=vector.source)

#Update database
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    if file.filename.endswith(".md"):
        docs = []       
        content = file.read().decode('utf-8')
        sections = extract_sections(content=content)
        for section in sections:
            new_doc = Document(
                page_content=section.strip(),
                metadata = {'source':UPLOAD_FOLDERs+'/'+file.filename})
            docs.append(new_doc)
            
        db= Chroma.from_documents(
                documents=docs, 
                embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                persist_directory="./vector/d4a1cccb-a9ae-43d1-8f1f-9919c90ad369"
                )
        db.persist()
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
        with open(UPLOAD_FOLDER+'/'+file.filename, 'w', encoding='utf-8') as save_file:
            save_file.write(content)
        return jsonify({'result':"update successful"})
        vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad369')
    else:
        return jsonify({'result':'No markdown file part in the request.'}), 400
        
#Download database
@app.get('/download')
def download_file(filename:str):

    return FileResponse(filename)

#Delete database
@app.post('/delete')
async def delete(item: dict):
    vector = vector_by_id('d4a1cccb-a9ae-43d1-8f1f-9919c90ad369')
    filename = item["filename"]
    os.remove(filename)
    print(filename)
    result = delete_by_file(vector,filename)
    return result

#Change model
@app.post('/model_choice')
async def model_choice(item: dict):
    global model_name_str 
    model_name_str = item['model_name']
    return {"result": 'success'}


@app.post('/query_history')
def query_history(query: str):
    global sql_conn
    cursor = sql_conn.cursor()
    cursor.execute(
        "SELECT query, model, answer, AVG(millisecs) FROM chat_history WHERE query = ? GROUP BY query, model, answer",
        (query,))

    data = cursor.fetchall()
    results = []
    for row in data:
        result_dict = {
            'query': row[0],
            'model': row[1],
            'answer': row[2],
            'time': row[3],
        }
        results.append(result_dict)

    cursor.close()
    return results


@app.get('/all_query_history')
def query_history():
    global sql_conn
    cursor = sql_conn.cursor()
    cursor.execute(
        "SELECT query, model, answer, AVG(millisecs) FROM chat_history GROUP BY query, model, answer")

    data = cursor.fetchall()
    results = []
    for row in data:
        result_dict = {
            'query': row[0],
            'model': row[1],
            'answer': row[2],
            'time': row[3],
        }
        results.append(result_dict)

    cursor.close()
    return results


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)
