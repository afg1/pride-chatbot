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

### Step3  Donwload the LLMs

#### Llama2:
- First, You should apply for a permission on [Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) and [Meta](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

- Second, followring the steps below:
```shell
# Login huggingface
huggingface-cli login

# Input your huggingface token
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

# Download the llama2 model
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```
#### Other LLMs
For the other LLMs, we do not need token verification. We could download them directly.

```shell
#chatglm2-6b
git clone https://huggingface.co/THUDM/chatglm2-6b

#Baichuan
git clone https://huggingface.co/baichuan-inc/Baichuan-7B

#GPT4ALL
wget https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin

# Vicuna
git clone https://huggingface.co/lmsys/vicuna-13b-v1.3

# mpt-7b-chat
git clone https://huggingface.co/mosaicml/mpt-7b-chat
```

### Step4  Donwload the Embedding Model (Sentence Transformers)
```shell

# Change the path from the folder `/root/autodl-tmp` saving LLMs to the folder `pride-chat-bot` for the project
cd pride-chat-bot

# Then download the sentence tranformers
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

### Step5 Install the requiremient
```shell
pip install -r requirements.txt
```

### Step6 LLMs Config
If you download the LLMs in a specific folder, you could update `config.yaml` config file accordingly.
```
# Two paths config for GPT4 ALL
# Path for saving model
GPT4ALL_PATH: /root/autodl-tmp
# Name of model file
GPTEALL_MODEL: ggml-gpt4all-j-v1.3-groovy.bin

# The other LLMs
chatglm2: /root/autodl-tmp/chatglm2-6b
Vicuna-13B: /root/autodl-tmp/vicuna-13b-v1.3
baichuan-7b: /root/autodl-tmp/Baichuan-7B
mbt-7b-chat: /root/autodl-tmp/mpt-7b-chat
llama2-chat: /root/autodl-tmp/Llama-2-7b-chat-hf
```

### Step7 Start Server
```
# You can change the URL and port number in line 327 of the server.py
python3 server.py

```
