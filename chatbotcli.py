
import platform

import yaml
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings  # Use to load the embedding model in hugginface
from langchain.vectorstores import Chroma  # A tool that converts documents into vectors and can store and read them
from langchain.prompts import PromptTemplate  # Tool for generating prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel  # tool for loading model from huggingface
import os

# Variables initialization
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def llm_model_init(model: str, gpu: bool) -> (AutoTokenizer, AutoModel):
    """
    Init model and tokenizer from provided path
    :param gpu: If use gpu to load model
    :param model_path: model path
    :return: tokenizer, model
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "ture"  # Load the environment variables required by the local model

    # Load the Tokenizer, convert the text input into an input that the model can accept
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # Load the model, load it to the GPU in half-precision mode
    if gpu:
        model = AutoModel.from_pretrained(model,trust_remote_code=True).half().cuda()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        model = AutoModel.from_pretrained(model, trust_remote_code=True).cpu().float()

    return tokenizer, model


# Load the specified private database (vector) by specifying the id
def vector_by_id(path_id: str, model: str) -> Chroma:
    # Set the path of the database
    directory = "./vector/" + path_id
    # Load private knowledge base, uses embedding model named sentence-transformers
    vector = Chroma(persist_directory=directory, embedding_function=HuggingFaceEmbeddings(model_name=model))
    return vector

# According to the query entered by the user, retrieve relevant documents in the private database (vector),
# and generate a complete prompt
def get_similar_answer(vector, query) -> str:
    # prompt template, you can add external strings to { }
    prompt_template = """
        You are a professional chat robot. Please answer the questions according to the following Knowledge, and please convert the language of the generated answer to the same language as the user
        ###Knowledge:{context}
        ###Question:{question}
        """
    # create prompt，assign variable that can be added from other str。
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


def build_prompt(history: list) -> str:
    prompt = "PRIDE ChatGLM-6B，clear to Clean the history, stop to exit the program"
    for query, response in history:
        prompt += f"\n\nQuery：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main(model, tokenizer):
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
        result = model.chat(tokenizer=tokenizer, input=prompt, history=history)
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
    tokenizer, model = llm_model_init(cfg['llm']['model'], cfg['llm']['gpu'])
    vector = vector_by_id(cfg['vector']['uui'], cfg['llm']['model'])
    main(model, tokenizer) # call main function
