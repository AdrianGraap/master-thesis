import json
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import numpy as np
import requests

PATH = './results/checkpoint-19955/'#
BATCH_SIZE = 10

DBPEDIA_SPOTLIGHT = 'https://api.dbpedia-spotlight.org/en/annotate'
HEADERS = {'Accept': 'application/json'}

#get label2id
with open('label2id_first_item.json', encoding='utf-8') as f:
    label2id = json.load(f)

id2label = {value: key for key, value in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForSequenceClassification.from_pretrained(PATH, id2label=id2label, label2id=label2id).to("cuda")

tokenizer_EL = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model_EL = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")


def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    ret[:] = (logits[:] >= 0).astype(int)
    return ret


data = []
prediction = []

with open('data_test.json', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

list_begin = 0

while True:
    input_texts = []
    ids = []

    for item in data[list_begin:list_begin+BATCH_SIZE]:
        answer = requests.get(DBPEDIA_SPOTLIGHT, params={'text': item['question']}, headers=HEADERS)
        answer_dict = json.loads(answer.text)
        if 'Resources' in answer_dict:
            for ent in answer_dict['Resources']:
                item['question_processed'] = item['question'].replace(ent['@surfaceForm'], '<ent>')
        if 'question_processed' in item:
            input_texts.append(item['question_processed'])
        else:
            input_texts.append(item['question'])
        ids.append(item['id'])

    # Encode the text
    encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cuda")

    # Call the model to predict under the format of logits of 27 classes
    logits = model(**encoded).logits.cpu().detach().numpy()


    # Decode the result
    preds = get_preds_from_logits(logits)
    decoded_preds = [[id2label[i] for i, l in enumerate(row) if l == 1] for row in preds]

    for text, pred, q_id in zip(input_texts, decoded_preds, ids):
        print(text)
        print("Label:", pred)
        print("")

        prediction.append({'question': text, 'relations': pred, 'id': q_id})

    list_begin += BATCH_SIZE
    if list_begin >= len(data):
        break

os.makedirs('eval', exist_ok=True)

path_gold = 'eval/gold.json'
with open(path_gold, 'w+', encoding='utf-8') as f:
    json.dump(data, f)

path_test = 'eval/test.json'
with open(path_test, 'w+', encoding='utf-8') as f:
    json.dump(prediction, f)
