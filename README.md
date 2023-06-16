# pride-chat-bot
This is the chatbot for Pride Team.
## Introduction
### Data preparation
- We need to create the embedding data according to our training target. In pride, we currently use the document in the [help page](https://www.ebi.ac.uk/pride/markdownpage/documentationpage)  for the training target. This embedding data will be stored in the server as files.

- The embedding priciple is that we split the documents in the `help page` by the `subtitle`. Therefore, each page will normally be divided into several segments. Each segment is the `string` format, then we will convert the `string` format to `document` format which could be accepted by the vector database like `Chroma`. We use `Sentence Transformer` to finish this step。

- In the end, we use the `api` in `Chorma` to save all these `document` into the vector database for future usage.

-  The specific (embedding) model for this step is called `all-MiniLM-L6-v2` and we need to load this model in this way below.
```python
#Load the specified private database (vector) by specifying the id
def vector_by_id(path_id: str):
    #Set the path of the database
    directory = "./vector/" + path_id
    #Load private knowledge base, uses embedding model named sentence-transformers
    vector = Chroma(persist_directory=directory, embedding_function=HuggingFaceEmbeddings(model_name="/hps/nobackup/juan/pride/chatbot/all-MiniLM-L6-v2"))
    return vector
```
- If we do not install `all-MiniLM-L6-v2` to the local machine, we could load the remote model by this way below. Only put the specific name as the `model_name`.

```python
 vector = Chroma(persist_directory=directory, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
```

### Chat

- User will send a question to the chatbot. The server side will receive this quesiton. We pass this quesiton to the vector database to dicide the first four most similar items according to the score. Then we will recover the vector data (document) to `string` again.

- Now we have the first four most similar string, we will compose a `template` which is made up by experience which will help LLM understand users' purpose more easily.

The template is like below: 

```
You should summerize the knowledge and provide concise answer.
Please answer the questions according following Knowledge, and please convert the language of the generated answer to the same language as the user.
If you does not know the answer to a question, please say I don’t know.
###Knowledge:{context}
###Question:{question}
```

The `question` part in the template is from user input. 

The `knowledge` part is from first four most similar string in the vector database. 

At last, we will send this `tempalte` into LLM and it will give us a summary which will be the answer for the users' question.

-  The specific (LLM) model for this step is called `chatglm-6b` and we need to load this model in this way below.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel  # tool for loading model from huggingface 

#Load the Tokenizer, convert the text input into an input that the model can accept
tokenizer = AutoTokenizer.from_pretrained("/hps/nobackup/juan/pride/chatbot/chatglm-6b", trust_remote_code=True)
#Load the model, load it to the GPU in half-precision mode
model = AutoModel.from_pretrained("/hps/nobackup/juan/pride/chatbot/chatglm-6b", trust_remote_code=True).half().cuda()
```

![Flow Chart](https://github.com/PRIDE-Archive/pride-chatbot/blob/main/flowchart.jpeg) 

## How to run this project
This project currently uses the [chatglm](https://github.com/THUDM/ChatGLM-6B) model and requires at least 12GB memory of GPU.
### Step1: Donwload the LLM
    #Make sure you have git-lfs installed (https://git-lfs.com)
    git lfs install
    git clone https://huggingface.co/THUDM/chatglm-6b
    #if you want to clone without large files – just their pointers
    #prepend your git clone with the following env var:
    GIT_LFS_SKIP_SMUDGE=1
### Step2: Donwload the Sentence Transformers
    git lfs install
    git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
### Step3: Install Dependecies
    pip install@ -r requirements.txt
### Step4: Specify the path to the local model in `main.python`
- The path of the local LLM
```python
#Load the Tokenizer, convert the text input into an input that the model can accept
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/chatglm-6b", trust_remote_code=True)
#Load the model, load it to the GPU in half-precision mode
model = AutoModel.from_pretrained("/root/autodl-tmp/chatglm-6b", trust_remote_code=True).half().cuda()  
```
- The path of the `Sentence Transformer` used for embedding
```python
#Load private knowledge base, uses embedding model named sentence-transformers
vector = Chroma(persist_directory=directory, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
 ```
### Step5: Start the server
```
python3 main.py
```
