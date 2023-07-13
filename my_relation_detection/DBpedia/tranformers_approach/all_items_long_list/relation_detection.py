import json
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import numpy as np
import requests

PATH = './results/checkpoint-19955/'#
BATCH_SIZE = 10

# DBPEDIA_SPOTLIGHT = 'https://api.dbpedia-spotlight.org/en/annotate'
# HEADERS = {'Accept': 'application/json'}

#get label2id
with open('label2id_first_item.json', encoding='utf-8') as f:
    label2id = json.load(f)

with open('../SMART2022-RL-dbpedia-relation-vocabulary.json', encoding='utf-8') as f:
    vocab = json.load(f)

num_labels = len(vocab)

id2label = {value: key for key, value in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForSequenceClassification.from_pretrained(PATH, id2label=id2label, label2id=label2id).to("cuda")

# tokenizer_EL = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
# model_EL = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")


def get_preds_from_logits(logits):
    return_value = []
    # (logits[0][i * num_labels:i * num_labels + num_labels] == np.amax(logits[0][i * num_labels:i * num_labels + num_labels])).astype(int)
    for counter, logit in enumerate(logits):
        ret = np.zeros(logit.shape)
        for i in range(3):
            if np.amax(logit[i * num_labels:i * num_labels + num_labels]) > 0:
                ret[i * num_labels:i * num_labels + num_labels] = (logit[i * num_labels:i * num_labels + num_labels] == np.amax(logit[i * num_labels:i * num_labels + num_labels])).astype(int)
        return_value.append(ret)
    return np.array(return_value)


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
        pred_after_mapping = [p.split('|')[0] for p in pred]

        if len(pred_after_mapping) > 1:
            print(text)
            print("Label:", pred_after_mapping)
            print("")

        prediction.append({'question': text, 'relations': pred_after_mapping, 'id': q_id})

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
