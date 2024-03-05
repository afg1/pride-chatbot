import time

import gspread
import pandas as pd
import requests
import json

import torch
from transformers import BertTokenizer, BertModel, LongformerTokenizer, LongformerModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Open the Google Spreadsheet using its title
gs = gspread.service_account()
spreadsheet = gs.open_by_url(
    'https://docs.google.com/spreadsheets/d/1LBRvDGUL5LrybuNu1EXfeaT8dg17pwTRfq28wtukkVc/edit?usp=sharing')  # Replace with the URL of your Google Sheet

# Select the first sheet
worksheet = spreadsheet.get_worksheet(0)

# Get the data as a list of dictionaries
data = worksheet.get_all_records()

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)

# Extract questions and gold standard answers
gold_standard_questions = df['Question'].tolist()
gold_standard_answers = df['Answer'].tolist()

# model_names = ["llama2-13b-chat", "chatglm3-6b", "open-hermes", "Mixtral"]
model_names = []

# Set the URL
url = 'https://www.ebi.ac.uk/pride/ws/archive/v2/chat'

# Set the headers
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,de;q=0.7,te;q=0.6',
    'Connection': 'keep-alive'
}

# List to store responses
responses = []

output_file_path = '../resources/responses18.tab'

# Iterate over model names
for model_name in model_names:
    print(f"\nModel: {model_name}\n")
    # Iterate over golden questions
    for question in gold_standard_questions:
        # Set the payload (data) for each question
        payload = {
            "Chat": {
                "model_name": model_name,
                "prompt": question.strip()
            }
        }
        json_payload = json.dumps(payload)
        response = requests.post(url, headers=headers, data=json_payload)

        json_response = response.json() if response.ok else f"Request failed with status code {response.status_code}"
        time_ms = json_response.get("timems") if response.ok else 0
        result = json_response.get("result").replace('\n', '\\n') \
            .replace('\t', '\\t') if response.ok else f"Request failed with status code {response.status_code}"

        if model_name == "llama2-13b-chat":
            time.sleep(10)

        with open(output_file_path, 'a') as file:
            file.write(f"{model_name}\t{question}\t{time_ms}\t{result.strip()}\n")

        # Append the response to the list
        responses.append((model_name, question, result, time_ms))

# Print the stored responses
for i, (model_name, question, response, time_ms) in enumerate(responses, start=1):
    print(f"Response {i} in {time_ms} for Model {model_name}  and Question '{question}': {response}")
    print("\n")

# Select the second sheet
worksheet = spreadsheet.get_worksheet(1)

# Get the data as a list of dictionaries
data = worksheet.get_all_records()

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)

response_chatglm = []
response_openhermes = []
response_mixtral = []
response_llama2 = []

for index, row in df.iterrows():
    if row['model-name'] == "chatglm3-6b":
        response_chatglm.append(row['response'])
    if row['model-name'] == "open-hermes":
        response_openhermes.append(row['response'])
    if row['model-name'] == "Mixtral":
        response_mixtral.append(row['response'])
    if row['model-name'] == "llama2-13b-chat":
        response_llama2.append(row['response'])

print(len(response_chatglm))

# Evaluate accuracy for each model

#
# # Load the pre-trained BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

model_name = 'allenai/longformer-base-4096'
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerModel.from_pretrained(model_name)


# Bert
# Chatglm Accuracy: 0.91
# OpenHermes Accuracy: 0.88
# Mixtral Accuracy: 0.93
# Llama2 Accuracy: 0.85

# Longform
# Chatglm Accuracy: 0.95
# OpenHermes Accuracy: 0.96
# Mixtral Accuracy: 0.97
# Llama2 Accuracy: 0.93


# def calculate_accuracy(model_responses, gold_standard):
#     # Tokenize and obtain embeddings for each sentence
#     embeddings_array1 = []
#     embeddings_array2 = []
#     input_token_sizes = []
#     for text in model_responses:
#         inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=False)
#         input_token_sizes.append(len(inputs['input_ids'][0]))
#         outputs = model(**inputs)
#         embeddings_array1.append(outputs.pooler_output.detach().numpy().flatten())
#
#     for text in gold_standard:
#         inputs = tokenizer(text, return_tensors='pt',max_length=512,truncation=False)
#         outputs = model(**inputs)
#         embeddings_array2.append(outputs.pooler_output.detach().numpy().flatten())
#
#     # Calculate cosine similarity
#     similarity_scores = []
#     for i in range((len(embeddings_array1))):
#         similarity_score = cosine_similarity([embeddings_array1[i]], [embeddings_array2[i]])[0][0]
#         similarity_scores.append(similarity_score)
#
#     return sum(similarity_scores) / len(similarity_scores), sum(input_token_sizes) / len(input_token_sizes)


def calculate_accuracy(model_responses, gold_standard):
    embeddings_array1 = []
    embeddings_array2 = []
    input_token_sizes = []
    word_counts = []
    for text in model_responses:
        inputs = tokenizer(text, return_tensors='pt')
        input_token_sizes.append(len(inputs['input_ids'][0]))
        word_counts.append(len(text.split(" ")))
        outputs = model(**inputs)
        with torch.no_grad():
            embeddings_array1.append(outputs.last_hidden_state.mean(dim=1).squeeze())

    for text in gold_standard:
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        with torch.no_grad():
            embeddings_array2.append(outputs.last_hidden_state.mean(dim=1).squeeze())

    # Calculate cosine similarity
    similarity_scores = []
    for i in range(len(embeddings_array1)):
        similarity_score = \
            cosine_similarity(torch.reshape(embeddings_array1[i], (1, -1)),
                              torch.reshape(embeddings_array2[i], (1, -1)))[
                0][0]
        similarity_scores.append(similarity_score)

    return sum(similarity_scores) / len(similarity_scores), sum(input_token_sizes) / len(input_token_sizes), sum(
        word_counts) / len(word_counts)


# def calculate_accuracy(model_responses, gold_standard):
#     similarity_scores = []
#     for model_text, gold_text in zip(model_responses,gold_standard):
#         # Tokenize the paragraphs
#         vectorizer = CountVectorizer().fit_transform([model_text, gold_text])
#         vectors = vectorizer.toarray()
#
#         # Calculate cosine similarity
#         cosine_sim = cosine_similarity(vectors)
#         similarity_score = cosine_sim[0, 1]
#         similarity_scores.append(similarity_score)
#     return sum(similarity_scores) / len(similarity_scores)


accuracy_model1, average_token_size1, average_word_size1 = calculate_accuracy(response_chatglm, gold_standard_answers)
accuracy_model2, average_token_size2, average_word_size2 = calculate_accuracy(response_openhermes, gold_standard_answers)
accuracy_model3, average_token_size3, average_word_size3 = calculate_accuracy(response_mixtral, gold_standard_answers)
accuracy_model4, average_token_size4, average_word_size4 = calculate_accuracy(response_llama2, gold_standard_answers)

# # Print individual accuracy scores
print(f"Chatglm Accuracy: {accuracy_model1:.2f} Average token size: {average_token_size1} Average word size: {average_word_size1}")
print(f"OpenHermes Accuracy: {accuracy_model2:.2f} Average token size: {average_token_size2} Average word size: {average_word_size2}")
print(f"Mixtral Accuracy: {accuracy_model3:.2f} Average token size: {average_token_size3} Average word size: {average_word_size3}")
print(f"Llama2 Accuracy: {accuracy_model4:.2f} Average token size: {average_token_size4} Average word size: {average_word_size4}")
