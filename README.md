# pride-chat-bot

This is the backend project for EBI chat-bot. 

The project could be deployed in a server with GPU based on the following steps.


### Step1 Install git-lfs
```shell
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
```

### Step2  Create a folder in the server to save LLM 
```shell
# Create a folder where the model parameters are stored
# such as /root/autodl-tmp
cd /root/autodl-tmp
```

### Step3  Download the LLMs

#### Llama2:
- First, You should apply for a permission on [Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) and [Meta](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

- Second, following the steps below:
```shell
# Login huggingface
huggingface-cli login

# Input your huggingface token
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

# Download the llama2 model
git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
```
#### Other LLMs
For the other LLMs, we do not need token verification. We could download them directly.

```shell
#chatglm3-6b
git clone https://huggingface.co/THUDM/chatglm3-6b

#Mistral-7B-Instruct-v0.2
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

#OpenHermes-2.5-Mistral-7B
git clone https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B


```

### Step4  Download the Embedding Model (Sentence Transformers)
```shell

# Change the path from the folder `/root/autodl-tmp` saving LLMs to the folder `pride-chat-bot` for the project
cd pride-chat-bot

# Then download the sentence transformers
git clone https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
```

### Step5 Install the requiremient
```shell
pip install -r requirements.txt
```

### Step6 LLMs Config
If you download the LLMs in a specific folder, you could update `config.yaml` config file accordingly.
```
# The other LLMs
chatglm3: /root/autodl-tmp/chatglm3-6b
Mistral-7B-Instruct-v0.2: /root/autodl-tmp/Mistral-7B-Instruct-v0.2
OpenHermes-2.5-Mistral-7B: /root/autodl-tmp/OpenHermes-2.5-Mistral-7B
llama2-chat: /root/autodl-tmp/Llama-2-13b-chat-hf
```

### Step7 Start Server
```
# You can change the URL and port number in line 327 of the server.py
python3 server.py

```
