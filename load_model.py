import os
import platform

import gpt4all
import torch
import torch.cuda as cuda
import transformers
import yaml
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, \
    AutoConfig, BitsAndBytesConfig  # tool for loading model from huggingface

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
    if choice == 'chatglm2-6b':  # load ChatGLM-6B
        model_path = cfg['llm']['chatglm2']
        # Load the Tokenizer, convert the text input into an input that the model can accept
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Load the model, load it to the GPU in half-precision mode
        if gpu:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cpu().float()
        return tokenizer, model
    elif choice == 'GPT4ALL':  # load the GPT4All only cpu
        model_path = cfg['llm']['GPT4ALL_PATH']
        model_name = cfg['llm']['GPTEALL_MODEL']
        model = gpt4all.GPT4All(model_path=model_path, model_name=model_name)
        return '1', model
    elif choice == 'vicuna-13b':  # load the vicuna-13b
        model_path = cfg['llm']['Vicuna-13B']
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True, device_map="auto",
            torch_dtype=torch.float16).to('cuda:0')
        return tokenizer, model
    elif choice == 'baichuan-7b':  # use baichaui-7b
        model_path = cfg['llm']['baichuan-7b']
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True,
                                                     torch_dtype=torch.float16).to('cuda:0')
        return tokenizer, model
    elif choice == 'mpt-7b':  # use mpt-7b
        model_path = cfg['llm']['mpt-7b-chat']
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # config.attn_config['attn_impl'] = 'triton'
        config.init_device = 'cuda:0'  # For fast initialization directly on GPU!
        config.max_seq_len = 789
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,  # Load model weights in bfloat16
            trust_remote_code=True
        )
        return tokenizer, model
    elif choice == 'llama2-chat':  # llama2-7b-chat
        llama2_path = cfg['llm']['llama2-chat']
        tokenizer = AutoTokenizer.from_pretrained(llama2_path, trust_remote_code=True, model_max_length=512)
        model = transformers.pipeline(
            "text-generation",
            model=llama2_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        return tokenizer, model
    elif choice == 'llama2-13b-chat':  # llama2-7b-chat
        llama2_path = cfg['llm']['llama2-13b-chat']
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(llama2_path, device_map="auto",
                                                          quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(llama2_path, trust_remote_code=True,truncation=True)
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
    elif choice == 'GPT4ALL':  # llama2-7b-chat
        model = gpt4all.GPT4All(
            model_path='/hps/nobackup/juan/pride/chatbot/pride-prd-chatbot/pride-new-chatbot/models/',
            model_name='ggml-gpt4all-j-v1.3-groovy.bin')
        tokenizer = None
        return tokenizer, model
    elif choice == "Mixtral":
        model_path = cfg["llm"]["Mixtral"]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                                          quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,truncation=True)
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
    if choice == 'chatglm2-6b':  # chat with ChatGLM
        result = model.chat(tokenizer, prompt, history=[])
        result = result[0]
    elif choice == 'vicuna-13b':  # chat with vicuna-13b
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
        outputs = model.generate(**inputs, max_new_tokens=256)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif choice == 'mpt-7b':  # chat with mpt-7b
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
        outputs = model.generate(**inputs, max_new_tokens=789)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif choice == 'llama2-chat':  # chat with llama2-chat
        # inputs = tokenizer(prompt,return_tensors="pt").to("cuda:0")
        out = model(
            prompt
        )
        print(out)
        # start_index = out[0]['generated_text'].find("###Questio:"+prompt)
        # content_start = start_index + len("###Question:"+prompt) + 1
        # end_index = out[0]['generated_text'].find("###", content_start)
        # result = out[0]['generated_text'][content_start:end_index].strip()
        result = out[0]['generated_text'].replace(prompt, "", 1).strip()
    elif choice == 'llama2-13b-chat' or choice == 'Mixtral':  # chat with llama2-chat
        # inputs = tokenizer(prompt,return_tensors="pt").to("cuda:0")
        out = model(
            prompt
        )
        # start_index = out[0]['generated_text'].find("###Questio:"+prompt)
        # content_start = start_index + len("###Question:"+prompt) + 1
        # end_index = out[0]['generated_text'].find("###", content_start)
        # result = out[0]['generated_text'][content_start:end_index].strip()
        result = out[0]['generated_text'].replace(prompt, "", 1).strip()
    elif choice == 'GPT4ALL':  # chat with mpt-7b
        messages = [{"role": "user", "content": prompt}]
        a = model.chat_completion(messages, default_prompt_header=False)
        result = a['choices'][0]['message']['content']
    elif choice == 'baichuan-7b':
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
        outputs = model.generate(**inputs, max_new_tokens=1024)
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(out)
        start_index = out.find("###Question:" + query)
        content_start = start_index + len("###Question:" + query) + 1
        end_index = out.find("###", content_start)
        result = out[content_start:end_index].strip()
        if len(result) == 0:
            result = 'model error'
    elif choice == "Mixtral":
        out = model(
            prompt
        )
        result = out[0]['generated_text'].replace(prompt, "", 1).strip()
    return result
