import gspread
import pandas as pd
import requests
import json

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Open the Google Spreadsheet using its title
gs = gspread.service_account()
spreadsheet = gs.open_by_url('https://docs.google.com/spreadsheets/d/1LBRvDGUL5LrybuNu1EXfeaT8dg17pwTRfq28wtukkVc/edit?usp=sharing')  # Replace with the URL of your Google Sheet

# Select the first sheet
worksheet = spreadsheet.get_worksheet(0)

# Get the data as a list of dictionaries
data = worksheet.get_all_records()

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)

# Extract questions and gold standard answers
gold_standard_questions = df['Question'].tolist()
gold_standard_answers = df['Answer'].tolist()
#["llama2-13b-chat", "chatglm3-6b", "open-hermes", "Mixtral"]
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

output_file_path = 'responses.csv'

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

        time_ms = json_response.get("timems")
        result = json_response.get("result").replace('\n', '\\n').replace('\t', '\\t')

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


# Evaluate accuracy for each model


# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and obtain embeddings for each sentence
embeddings_array1 = []
embeddings_array2 = []


def calculate_accuracy(mode1_responses, gold_standard):
    for text in mode1_responses:
        inputs = tokenizer(text, return_tensors='pt', max_length=2200, truncation=True)
        outputs = model(**inputs)
        embeddings_array1.append(outputs.pooler_output.detach().numpy().flatten())

    for text in gold_standard:
        inputs = tokenizer(text, return_tensors='pt', max_length=2200, truncation=True)
        outputs = model(**inputs)
        embeddings_array2.append(outputs.pooler_output.detach().numpy().flatten())

    # Calculate cosine similarity
    similarity_scores = []
    for i in range(len(mode1_responses)):
        similarity_score = cosine_similarity([embeddings_array1[i]], [embeddings_array2[i]])[0][0]
        similarity_scores.append(similarity_score)

    return sum(similarity_scores)/len(similarity_scores)


accuracy_model1 = calculate_accuracy(response_chatglm, gold_standard_answers)
accuracy_model2 = calculate_accuracy(response_openhermes, gold_standard_answers)
accuracy_model3 = calculate_accuracy(response_mixtral, gold_standard_answers)
accuracy_model4 = calculate_accuracy(response_llama2, gold_standard_answers)

# Print individual accuracy scores
print(f"Model 1 Accuracy: {accuracy_model1:.4f}")
print(f"Model 2 Accuracy: {accuracy_model2:.4f}")
print(f"Model 3 Accuracy: {accuracy_model3:.4f}")
print(f"Model 4 Accuracy: {accuracy_model4:.4f}")