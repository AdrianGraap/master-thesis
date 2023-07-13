import json

if __name__ == '__main__':
    with open('../results_zwischenergebnisse/lcquad_earl_results.json', encoding='utf-8') as f:
        earl = json.load(f)

    with open('../results_zwischenergebnisse/lcquad_falcon_results.json', encoding='utf-8') as f:
        falcon = json.load(f)

    with open('../datasets/LCQuad.json', encoding='utf-8') as f:
        lcquad = json.load(f)

    for earl_set, falcon_set, lcquad_set in zip(earl, falcon, lcquad):
        earl_set['true_predicates_corrected'] = lcquad_set['true_relations']
        falcon_set['true_predicates_corrected'] = lcquad_set['true_relations']

    with open('../results_zwischenergebnisse/LCQuad_EARL_corrected.json', 'w', encoding='utf-8') as json_file:
        json.dump(earl, json_file, ensure_ascii=False, indent=4)

    with open('../results_zwischenergebnisse/LCQuad_FALCON_corrected.json', 'w', encoding='utf-8') as json_file:
        json.dump(falcon, json_file, ensure_ascii=False, indent=4)