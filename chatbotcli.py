
import platform
import gpt4all
import yaml
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings  # Use to load the embedding model in hugginface
from langchain.vectorstores import Chroma  # A tool that converts documents into vectors and can store and read them
from langchain.prompts import PromptTemplate  # Tool for generating prompts
from transformers import AutoTokenizer, AutoModel  # tool for loading model from huggingface
import os

# Variables initialization
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False
os.environ["TOKENIZERS_PARALLELISM"] = "ture"  # Load the environment variables required by the local model

def llm_model_init(choice: str, gpu: bool) -> (AutoTokenizer, AutoModel):
    """
    Init model and tokenizer from provided path
    :param gpu: If use gpu to load model
    :param model_path: model path
    :return: tokenizer, model
    """
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    if choice =='1':  #load ChatGLM-6B
        model_path = cfg['llm']['chatglm']
        # Load the Tokenizer, convert the text input into an input that the model can accept
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Load the model, load it to the GPU in half-precision mode
        if gpu:
            model = AutoModel.from_pretrained(model,trust_remote_code=True).half().cuda()
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            model = AutoModel.from_pretrained(model, trust_remote_code=True).cpu().float()
        return tokenizer, model
    elif choice == '2': #load the GPT4All only cpu
        model_path = cfg['llm']['GPT4ALL_PATH']
        model_name = cfg['llm']['GPTEALL_MODEL']
        model = gpt4all.GPT4All(model_path=model_path, model_name=model_name)
        return '1', model

#chat with model    
def llm_chat(choice:str,prompt:str,tokenizer,model):
    if choice =='1':#chat with ChatGLM
        result = model.chat(tokenizer,prompt, history=[])
    elif choice =='2':#chat with GPT4ALL 
        messages = [{"role": "user", "content": prompt}]
        result = model.chat_completion(messages,default_prompt_header=False)
    return result

# Load the specified private database (vector) by specifying the id
def vector_by_id(database_path: str, model: str) -> Chroma:
    # Set the path of the database
    directory = database_path
    isExist = os.path.exists(directory)
    print("Vector database {} exist: {}".format(directory, isExist))
    # Load private knowledge base, uses embedding model named sentence-transformers
    vector = Chroma(persist_directory=directory, embedding_function=HuggingFaceEmbeddings(model_name=model))
    return vector

# According to the query entered by the user, retrieve relevant documents in the private database (vector),
# and generate a complete prompt
def get_similar_answer(vector, query) -> str:
    #prompt template, you can add external strings to { }
    prompt_template = """
    You should summerize the knowledge and provide concise answer
    Please answer the questions according following Knowledge, and please convert the language of the generated answer to the same language as the user.
    If you does not know the answer to a question, please say I don’t know.
    ###Knowledge:{context}
    ###Question:{question}

    """
    #create prompt，assign variable that can be add from other str。
    #question：input from user。
    #context：knowledge search in the database according to the question from user
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    #loading retriever in database,Search for the top three most similar document fragments
    #retriever = vector.as_retriever(search_kwargs={"k": 3})
    #Searching in the database according to input from user,Retu rns relevant document and similarity score
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



# def get_similar_answer(vector, query) -> str:
#     prompt_template = """
#     ###Knowledge:{context}
#     ###Question:{question}
#     """
#     # create prompt，assign variable that can be add from other str。
#     # question：input from user。
#     # context：knowledge search in the database according to the question from user
#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )

#     # loading retriever in database,Search for the top three most similar document fragments
#     retriever = vector.as_retriever(search_kwargs={"k": 3})

#     # Searching in the database according to input from user,Returns relevant document and similarity score
#     docs = retriever.get_relevant_documents(query=query)
#     # put the relevant document into context
#     context = [d.page_content for d in docs]

#     # add the question input by user ande the relevant into prompt
#     result = prompt.format(context="\n".join(context), question=query)
#     return result


def build_prompt(history: list) -> str:
    prompt = "PRIDE ChatGLM-6B，clear to Clean the history, stop to exit the program"
    for query, response in history:
        prompt += f"\n\nQuery：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main(choice:str, model, tokenizer, vector):
    history = []
    global stop_stream
    print("PRIDE Chatbot，clear to Clean the history, stop to exit the program")
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
        prompt,docs = get_similar_answer(vector, query)
        print(docs)
        result = llm_chat(choice, prompt, tokenizer, model)
        print('Answer:')
        if choice =='1':
            print(result[0])
        else:
            print(result)


        # for response, history in result:
        #     if stop_stream:
        #         stop_stream = False
        #         break
        #     else:
        #         count += 1
        #         if count % 8 == 0:
        #             os.system(clear_command)
        #             print(build_prompt(history), flush=True)
        #             signal.signal(signal.SIGINT, signal_handler)
        # os.system(clear_command)
        # print(build_prompt(history), flush=True)

# main function
if __name__ == '__main__':
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    for llm in cfg:
        print(llm)
    print("Please choose the model!\n 1 - chatglm \n 2 - GPT4ALL")
    choice = input('\nChoice:') #user choose model with number 
    tokenizer, model = llm_model_init(choice.strip(), cfg['llm']['gpu'])
    database_path = cfg['vector']['cli_store'] + cfg['vector']['uui'] 
    vector = vector_by_id(database_path=database_path,model=cfg['llm']['embedding'])
    main(choice, model, tokenizer, vector) # call main function
