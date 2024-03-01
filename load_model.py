import os
import platform

import torch
import torch.cuda as cuda
import transformers
import yaml
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, \
    BitsAndBytesConfig  # tool for loading model from huggingface

# Variables initialization
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False
os.environ["TOKENIZERS_PARALLELISM"] = "ture"  # Load the environment variables required by the local model


def llm_model_init(choice: str, gpu: bool):
    torch.cuda.empty_cache()
    cuda.empty_cache()
    """
    Init model and tokenizer from provided path
    :param gpu: If use gpu to load model
    :param model_path: model path
    :return: tokenizer, model
    """
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    if choice == 'chatglm3-6b':  # load ChatGLM-6B
        model_path = cfg['llm']['chatglm3']
        # Load the Tokenizer, convert the text input into an input that the model can accept
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Load the model, load it to the GPU in half-precision mode
        if gpu:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cpu().float()
        return tokenizer, model
    elif choice == 'llama2-13b-chat' or choice == 'Mixtral' or choice == 'open-hermes':
        if choice == 'llama2-13b-chat':
            model_path = cfg['llm']['llama2-13b-chat']
        elif choice == 'Mixtral':
            model_path = cfg["llm"]["Mixtral"]
        elif choice == 'open-hermes':
            model_path = cfg["llm"]["open-hermes"]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                                          quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, truncation=True,
                                                  model_max_length=1024)
        model = transformers.pipeline(
            "text-generation",
            model=model_4bit,
            tokenizer=tokenizer,
            use_cache=True,
            device_map="auto",
            truncation=True,
            max_length=1200,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer, model


# chat with model
def llm_chat(choice: str, prompt: str, tokenizer, model, query: str):
    if choice == 'chatglm3-6b':  # chat with ChatGLM
        result = model.chat(tokenizer, prompt, history=[])
        result = result[0]
    elif choice == 'llama2-13b-chat' or choice == 'Mixtral' or choice == 'open-hermes':
        # inputs = tokenizer(prompt,return_tensors="pt").to("cuda:0")
        out = model(
            prompt,
            truncation=True
        )
        # start_index = out[0]['generated_text'].find("###Questio:"+prompt)
        # content_start = start_index + len("###Question:"+prompt) + 1
        # end_index = out[0]['generated_text'].find("###", content_start)
        # result = out[0]['generated_text'][content_start:end_index].strip()
        result = out[0]['generated_text'].replace(prompt, "", 1).strip()
    return result
