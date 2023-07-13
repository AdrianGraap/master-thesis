import json

if __name__ == '__main__':

    with open('../results_zwischenergebnisse/qald9_falcon_results.json', encoding='utf-8') as f:
        old = json.load(f)

    with open('../results_zwischenergebnisse/qald9_falcon_results_no_request.json', encoding='utf-8') as f:
        new = json.load(f)

    for old_set, new_set in zip(old, new):
        old_set['true_predicates_wikidata'] = new_set['true_predicates_wikidata']

    with open('../results_zwischenergebnisse/QALD9_FALCON.json', 'w', encoding='utf-8') as outfile:
        json.dump(old, outfile, ensure_ascii=False, indent=4)