import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

PATH = './results/checkpoint-19955/'#
BATCH_SIZE = 10

#get label2id
with open('label2id.json', encoding='utf-8') as f:
    label2id = json.load(f)

id2label = {value: key for key, value in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForSequenceClassification.from_pretrained(PATH, id2label=id2label, label2id=label2id).to("cuda")

def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    ret[:] = (logits[:] >= 0).astype(int)
    return ret

list_begin = 0
data = []
prediction = []

with open('data_test.json', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

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
        print(text)
        print("Label:", pred)
        print("")

        predictions = []

        for x in pred:
            predictions.append(x.split('|')[0])

        if pred:
            prediction.append({'question': text, 'relations': predictions, 'id': q_id})
        else:
            prediction.append({'question': text, 'relations': [], 'id': q_id})

    list_begin += BATCH_SIZE
    if list_begin >= len(data):
        break

with open('eval/gold.json', 'w', encoding='utf-8') as f:
    json.dump(data, f)

with open('eval/test.json', 'w', encoding='utf-8') as f:
    json.dump(prediction, f)
