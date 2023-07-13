import pandas as pd
import json

lcquad_earl = '../results/lcquad_earl_results.json'
qald_earl = 'qald_earl_results.json'
simple_earl = 'simple_questions_earl_results.json'
qald_9 = 'qald9_earl_results.json'
qald_9_no_request = '../qald9_earl_results_no_request.json'

filename = lcquad_earl

if __name__ == '__main__':
    with open(filename, 'rb') as file:
        data = json.load(file)

    for item in data:
        new_relations = []
        for relation in item['relations']:
            new_relations.append(relation[0][1].split('/')[-1])
        new_entities = []
        for entity in item['entities']:
            new_entities.append(entity[0][1])
        item['pred_entities'] = new_entities
        item['pred_relations'] = new_relations

        new_relations = []
        for relation in item['true_predicates']:
            new_relations.append(relation.split('/')[-1])
        item['true_predicates'] = new_relations

    if filename == qald_9 or filename == qald_9_no_request:
        dataframe = pd.DataFrame(data,
                                 columns=["question", "sparql_query", "true_entities_wikidata", "true_entities_dbpedia",
                                          "pred_entities", "true_predicates_wikidata", "true_predicates_dbpedia",
                                          "pred_relations"])
    else:
        dataframe = pd.DataFrame(data, columns=["question", "sparql_query", "true_entities", "pred_entities", "true_predicates", "pred_relations"])

    filename = filename.split('.')[0]

    dataframe.to_csv(f'../csv_results/{filename}_only_last_party.csv', sep='|')
