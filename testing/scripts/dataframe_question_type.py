import json
import pandas as pd
import sys

sys.path.append('..')

from analysis.scripts.get_question_type import check

if __name__ == '__main__':
    with open('../datasets/LCQuad.json', encoding='utf-8') as f:
        lcquad = json.load(f)

    with open('../datasets/qald_9_plus_test_dbpedia.json', encoding='utf-8') as f:
        qald = json.load(f)

    quest_list = []
    query_list = []
    dataset_list = []
    question_type_list = []
    for item in lcquad:
        quest_list.append(item['question'])
        query_list.append(item['sparql_query'])
        dataset_list.append(('lcquad'))
        question_type_list.append(check(item['question'], item['sparql_query']))

    for item in qald['questions']:
        quest_list.append(item['question'][0]['string'])
        query_list.append(item['query']['sparql'])
        dataset_list.append(('qald'))
        question_type_list.append(check(item['question'][0]['string'], item['query']['sparql']))

    df = pd.DataFrame(list(zip(quest_list, query_list, dataset_list, question_type_list)),
                      columns=['question', 'query', 'dataset', 'question_type'])

    df.to_csv('../results/question_type.csv')
