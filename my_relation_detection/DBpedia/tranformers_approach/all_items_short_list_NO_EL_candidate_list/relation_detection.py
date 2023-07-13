import json
import os
from candidate_list.my_own_candidate_list_scripts.get_candidates import get_candidates_single_question

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

PATH = './results/checkpoint-19955/'#
BATCH_SIZE = 10

#get label2id
with open('label2id_first_item.json', encoding='utf-8') as f:
    label2id = json.load(f)

id2label = {value: key for key, value in label2id.items()}
vocab = [key for key, value in label2id.items()]

tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForSequenceClassification.from_pretrained(PATH, id2label=id2label, label2id=label2id).to("cuda")


def get_candidate_array(candidate):
    candidate_array = np.zeros(len(vocab))
    common_items = list(set(candidate).intersection(list(vocab)))
    common_ids = [value for key, value in label2id.items() if key in common_items]
    if not common_ids:
        return np.ones(len(vocab))
    for c_id in common_ids:
        candidate_array[c_id] = 1
    return candidate_array



def get_preds_from_logits(logits, candidate_list, candidate_lens):
    # TODO: Kandidatenliste abgleichen
    # Idee: Array erstellen, was an der n-ten Stelle (n = id aus label2id) 0 ist, wenn Kandidat NICHT in label2id
    # existiert, 1 wenn Kandidat existiert.
    ret = np.zeros(logits.shape)
    for i, r, logit, candidate, length in zip(range(len(logits)), ret, logits, candidate_list, candidate_lens):
        new_ret = np.zeros(r.shape)
        if length != 0:
            n = length
        else:
            n = np.count_nonzero(logit >= 0)
        if n == 0:
            ret[i] = new_ret.astype(int)
        else:
            candidate_array = get_candidate_array(candidate)
            r = logit * candidate_array
            r[r == 0] = np.NINF
            index_array = np.argsort(r)[-n:]         # n größte indexe
            for x in index_array:
                new_ret[x] = 1
            ret[i] = new_ret.astype(int)
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
    candidate_lens = []
    all_candidates = []

    for item in data[list_begin:list_begin+BATCH_SIZE]:
        has_candidates = False
        candidates = []
        ent_dict: dict = get_candidates_single_question(item['question'], 'dbpedia', 'spacy') # TODO: nur für vorgefertigte Liste so gemacht, bei echter Anwendung diese Zeile wieder entkommentieren
        for key, value in ent_dict.items():
            candidates.extend(value)
            has_candidates = True
        if has_candidates:
            all_candidates.append(candidates)
        else:
            all_candidates.append([])
        input_texts.append(item['question'])
        ids.append(item['id'])
        candidate_lens.append(len(ent_dict))

    # Encode the text
    encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cuda")

    # Call the model to predict under the format of logits of 27 classes
    logits = model(**encoded).logits.cpu().detach().numpy()


    # Decode the result
    preds = get_preds_from_logits(logits, all_candidates, candidate_lens)
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

with open('eval/gold.json', 'w', encoding='utf-8') as f:
    json.dump(data, f)

with open('eval/test.json', 'w', encoding='utf-8') as f:
    json.dump(prediction, f)
