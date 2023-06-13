# import KonwledgeBaseQa  # class for search knowledge in database
from langchain.chains import LLMChain  # A tool which use LLM in langchain
from langchain.llms import HuggingFacePipeline # a tool load model in huggingface by pipline（vicuna）
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings # Use to load the embedding model in hugginface
from langchain.vectorstores import Chroma  #A tool that converts documents into vectors and can store and read them
from langchain.docstore.document import Document # a function use to change chunk of the document into a format acceptable to chroma
from langchain.prompts import PromptTemplate  # Tool for generating prompts
# from langchain.memory import ConversationBufferMemory  #tool for loading conversation history in prompt
# from langchain.chains import ConversationalRetrievalChain #tool for loading conversation history in prompt
from flask import Flask, request, jsonify #flask is a web server framework
from flask_cors import CORS # allow cross domain request
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel  # tool for loading model from huggingface 
import os # operating system
os.environ["TOKENIZERS_PARALLELISM"] = "ture"  #Load the environment variables required by the local model


#Load the Tokenizer, convert the text input into an input that the model can accept
tokenizer = AutoTokenizer.from_pretrained("/hps/nobackup/juan/pride/chatbot/chatglm-6b", trust_remote_code=True)
#Load the model, load it to the GPU in half-precision mode
model = AutoModel.from_pretrained("/hps/nobackup/juan/pride/chatbot/chatglm-6b", trust_remote_code=True).half().cuda()  



#Load the specified private database (vector) by specifying the id
def vector_by_id(path_id: str):
    #Set the path of the database
    directory = "./vector/" + path_id
    #Load private knowledge base, uses embedding model named sentence-transformers
    vector = Chroma(persist_directory=directory, embedding_function=HuggingFaceEmbeddings(model_name="/hps/nobackup/juan/pride/chatbot/all-MiniLM-L6-v2"))
    return vector

#According to the query entered by the user, retrieve relevant documents in the private database (vector), and generate a complete prompt
def get_similar_answer(vector, query):
        #prompt template, you can add external strings to { }
        prompt_template = """
        You are a professional chat robot. Please answer the questions according to the following Knowledge, and please convert the language of the generated answer to the same language as the user
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
        retriever = vector.as_retriever(search_kwargs={"k": 3})
        
        #Searching in the database according to input from user,Returns relevant document and similarity score
        docs = retriever.get_relevant_documents(query=query)
        #put the relevant document into context
        context = [d.page_content for d in docs] 
        
       
        #add the question input by user ande the relevant into prompt
        result = prompt.format(context="\n".join(context), question=query)
        return result

#load the database according to uuid
vector = vector_by_id("d4a1cccb-a9ae-43d1-8f1f-9919c90ad369")
#create an instance of the Flask class
app = Flask(__name__)
#allowing cross-origin requests to access API endpoints. 
CORS(app)
# enable debug mode
app.config['DEBUG'] = True
# create route
@app.route('/post-file', methods=['POST'])

def post_file():
    #Parse json variables from frontend
    data = request.get_json()
    # extract user input
    query = data['query']
    #Retrieve relevant documents in databse and form a prompt
    prompt = get_similar_answer(vector,query)
    #Input the prompt into the model
    result= model.chat(tokenizer = tokenizer, input = prompt, history=[])
    #Convert the result into json format and return to the foreground
    return jsonify({'ans':result[0]})

#main function
if __name__ == '__main__':
    
    #start service
    app.run( host = '0.0.0.0',port=6006, debug=True,use_reloader=False)
   