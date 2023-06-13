# import KonwledgeBaseQa  # class for search knowledge in database
import platform
import signal

from langchain.chains import LLMChain  # A tool which use LLM in langchain
from langchain.llms import HuggingFacePipeline  # a tool load model in huggingface by pipline（vicuna）
from langchain.embeddings import HuggingFaceEmbeddings, \
    SentenceTransformerEmbeddings  # Use to load the embedding model in hugginface
from langchain.vectorstores import Chroma  # A tool that converts documents into vectors and can store and read them
from langchain.docstore.document import \
    Document  # a function use to change chunk of the document into a format acceptable to chroma
from langchain.prompts import PromptTemplate  # Tool for generating prompts
# from langchain.memory import ConversationBufferMemory  #tool for loading conversation history in prompt
# from langchain.chains import ConversationalRetrievalChain #tool for loading conversation history in prompt
from flask import Flask, request, jsonify  # flask is a web server framework
from flask_cors import CORS  # allow cross domain request
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel  # tool for loading model from huggingface
import os  # operating system

os.environ["TOKENIZERS_PARALLELISM"] = "ture"  # Load the environment variables required by the local model

# Load the Tokenizer, convert the text input into an input that the model can accept
tokenizer = AutoTokenizer.from_pretrained("/hps/nobackup/juan/pride/chatbot/chatglm-6b", trust_remote_code=True)
# Load the model, load it to the GPU in half-precision mode
model = AutoModel.from_pretrained("/hps/nobackup/juan/pride/chatbot/chatglm-6b", trust_remote_code=True).half().cuda()


# Load the specified private database (vector) by specifying the id
def vector_by_id(path_id: str):
    # Set the path of the database
    directory = "./vector/" + path_id
    # Load private knowledge base, uses embedding model named sentence-transformers
    vector = Chroma(persist_directory=directory, embedding_function=HuggingFaceEmbeddings(
        model_name="/hps/nobackup/juan/pride/chatbot/all-MiniLM-L6-v2"))
    return vector

# According to the query entered by the user, retrieve relevant documents in the private database (vector), and generate a complete prompt
def get_similar_answer(vector, query):
    # prompt template, you can add external strings to { }
    prompt_template = """
        You are a professional chat robot. Please answer the questions according to the following Knowledge, and please convert the language of the generated answer to the same language as the user
        ###Knowledge:{context}
        ###Question:{question}
        """
    # create prompt，assign variable that can be add from other str。
    # question：input from user。
    # context：knowledge search in the database according to the question from user
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # loading retriever in database,Search for the top three most similar document fragments
    retriever = vector.as_retriever(search_kwargs={"k": 3})

    # Searching in the database according to input from user,Returns relevant document and similarity score
    docs = retriever.get_relevant_documents(query=query)
    # put the relevant document into context
    context = [d.page_content for d in docs]

    # add the question input by user ande the relevant into prompt
    result = prompt.format(context="\n".join(context), question=query)
    return result


# load the database according to uuid
vector = vector_by_id("d4a1cccb-a9ae-43d1-8f1f-9919c90ad369")
# create an instance of the Flask class

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

def build_prompt(history):
    prompt = "PRIDE ChatGLM-6B，clear to Clean the history, stop to exit the program"
    for query, response in history:
        prompt += f"\n\nQuery：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

def main():
    history = []
    global stop_stream
    print("PRIDE ChatGLM-6B，clear to Clean the history, stop to exit the program")
    while True:
        query = input("\nQuestion：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("PRIDE ChatGLM-6B，clear to Clean the history, stop to exit the program")
            continue
        count = 0
        prompt = get_similar_answer(vector, query)
        for response, history in model.stream_chat(prompt, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


# main function
if __name__ == '__main__':
    main()
