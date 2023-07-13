import json

if __name__ == '__main__':
    n = 3
    corr = 0
    ges = 0
    recall_list = []

    lcquad = '../results/lcquad_earl_results_NEW_MODEL.json'
    qald = '../results/qald9_earl_results_NEW_MODEL.json'

    dataset = qald

    if dataset == lcquad:
        key = 'true_relations'
    elif dataset == qald:
        key = 'true_predicates_dbpedia'

    with open(dataset, encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        rec_corr = 0
        rec_ges = 0
        predicted = item['relations']
        true_relations = item[key]

        for rel in true_relations:
            remove = False
            for pred_list in predicted:
                check_list = [el[1] for el in pred_list]
                if len(check_list) < n:
                    lim = len(check_list)
                else:
                    lim = n
                if rel in check_list[:lim]:
                    corr += 1
                    ges += 1
                    rec_corr += 1
                    rec_ges += 1
                    remove_list = pred_list
                    remove = True
                    break
            if remove:
                predicted.remove(remove_list)
            else:
                ges += 1
                rec_ges += 1
        recall_list.append(rec_corr/rec_ges)
    print(corr/ges)
    print(sum(recall_list)/len(recall_list))
