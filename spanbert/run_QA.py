"""
This script is used for running models from huggingface on the test SQUaD dataset.
Change the model name as required, as well as the target path to save the .json results.
The json results can be directly used in the official SQUaD evaluate-v2.0.py script.
"""

from transformers import pipeline
import json

question_answerer = pipeline("question-answering",
                             model="JMatthewChiam/4248-spanBERT-Base",
                             device=0)

count = 0
predictions = {}

with open('dev-v1.1.json', 'r') as file:
    squad_data = json.load(file)

for item in squad_data['data']:
    for paragraph in item['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question_id = qa['id']
            question = qa['question']

            answer = question_answerer(question=question, context=context)
            predictions[question_id] = answer['answer']

            count += 1
            print(f"{count} of 10570")

# change output file name as needed
with open('predictions_spanBERT_model11.json', 'w') as output_file:
    json.dump(predictions, output_file, indent=1)
