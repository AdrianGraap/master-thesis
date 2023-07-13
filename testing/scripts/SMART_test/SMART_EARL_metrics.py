import json
from collections import Counter
import nltk
import pandas as pd

punctuation_list = [',', '.', '?']

def compute_metrics_earl(true_rel, pred_rel):
    len_pred = len(pred_rel)
    len_true = len(true_rel)
    corr = 0

    for pred in pred_rel:
        remove = False
        x = pred.replace('http://dbpedia.org/ontology/', 'dbo:').replace('http://dbpedia.org/property/', 'dbp:')
        for true in true_rel:
            if x in true:
                corr += 1
                item_to_delete = true
                remove = True
                break
        if remove:
            true_rel.remove(item_to_delete)
    if corr == 0:
        return 0, 0, 0
    prec = corr / len_pred
    rec = corr / len_true
    f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


if __name__ == "__main__":
    all_relations = []
    question_list = []
    relation_count_list = []
    gold_standard_list = []
    predicted_relation_list = []
    word_count_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    question_type_list = []

    with open('../../results/SMART2022_EARL.json', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        true_relations = [x for x in item['relations']]
        predicted_relations = [x[0][1] for x in item['predicted_relations']]

        question_list.append(item['question'])
        word_count_list.append(len([i for i in nltk.word_tokenize(item['question'])
                                    if i not in punctuation_list]))
        relation_count_list.append(item['num_of_rels'])
        gold_standard_list.append('|||'.join(['|'.join(x) for x in true_relations]))
        predicted_relation_list.append('|'.join(predicted_relations))
        question_type_list.append(item['question_type'])

        precision, recall, f1score = compute_metrics_earl(true_relations, predicted_relations)
        # s_precision, s_recall, s_f1score = compute_metrics_earl_superclass(item, key, system_configs)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1score)
    df = pd.DataFrame(list(zip(question_list, relation_count_list, gold_standard_list, predicted_relation_list,
                               word_count_list, question_type_list, precision_list, recall_list, f1_list)),
                      columns=['question', 'relation_count', 'true_relations', 'predicted_relation', 'word_count',
                               'question_type', 'precision', 'recall', 'f1'])
    df.to_csv('csv/SMART_EARL.csv')
