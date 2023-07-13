import json

if __name__ == '__main__':

    with open('../results_zwischenergebnisse/qald9_earl_results.json', encoding='utf-8') as f:
        old = json.load(f)

    with open('../datasets/qald_9_plus_test_dbpedia.json', encoding='utf-8') as f:
        new = json.load(f)

    for old_set, new_set in zip(old, new["questions"]):
        old_set['true_predicates_dbpedia'] = new_set['true_relations']

    with open('../results_zwischenergebnisse/qald9_earl_results_CORRECTED.json', 'w', encoding='utf-8') as outfile:
        json.dump(old, outfile, ensure_ascii=False, indent=4)