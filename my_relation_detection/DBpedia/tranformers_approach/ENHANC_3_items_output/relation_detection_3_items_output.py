import json
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

PATH = '../all_items_short_list/results/checkpoint-19955/'#
BATCH_SIZE = 10


def namespace_mapping(pred):
    with open('../SMART2022-RL-dbpedia-relation-vocabulary.json', encoding='utf-8') as f:
        vocab = json.load(f)
    if 'dbo:' in pred:
        new_pred = pred.replace('dbo:', 'dbp:')
    elif 'dbp:' in pred:
        new_pred = pred.replace('dbp:', 'dbo:')
    else:
        return pred, ''

    if new_pred in vocab:
        return pred, new_pred
    else:
        return pred, ''


def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    for i, logit in enumerate(logits):
        l = np.zeros(logit.shape)
        ind = np.argpartition(logit, -3)[-3:]
        for index in ind:
            l[index] = 1
        ret[i] = l
        # if np.any((logit[:] >= 0).astype(int)):
        #     ret[i] = (logit[:] >= 0).astype(int)
        # else:
        #     ret[i] = (logit[:] == np.max(logit)).astype(int)
    return ret

def main():
    # get label2id
    with open('../all_items_short_list/label2id_first_item.json', encoding='utf-8') as f:
        label2id = json.load(f)

    id2label = {value: key for key, value in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForSequenceClassification.from_pretrained(PATH, id2label=id2label, label2id=label2id).to("cuda")

    data = []
    prediction = []

    with open('../all_items_short_list/data_test.json', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    list_begin = 0

    while True:
        input_texts = []
        ids = []

        for item in data[list_begin:list_begin + BATCH_SIZE]:
            input_texts.append(item['question'])
            ids.append(item['id'])

        # Encode the text
        encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to(
            "cuda")

        # Call the model to predict under the format of logits of 27 classes
        logits = model(**encoded).logits.cpu().detach().numpy()

        # Decode the result
        preds = get_preds_from_logits(logits)
        decoded_preds = [[id2label[i] for i, l in enumerate(row) if l == 1] for row in preds]

        for text, pred_list, q_id in zip(input_texts, decoded_preds, ids):
            all_preds = []
            print(text)
            print("Label:", pred_list)
            print("")

            prediction.append({'question': text, 'relations': pred_list, 'id': q_id})

        list_begin += BATCH_SIZE
        if list_begin >= len(data):
            break

    os.makedirs('eval', exist_ok=True)

    with open('eval/gold.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)

    with open('eval/test.json', 'w', encoding='utf-8') as f:
        json.dump(prediction, f)


if __name__ == '__main__':
    main()
