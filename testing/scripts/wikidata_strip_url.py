import json
import re
import requests
import pickle
import pandas as pd

lcquad = '../lcquad_falcon_results.json'
qald = '../qald_falcon_results.json'
simple = '../simple_questions_falcon_results.json'
qald9 = '../qald9_falcon_results.json'


def get_relation_code(relation):
    relation_codes = []
    wikidata_regex = r'P[0-9]+'
    matches = re.finditer(wikidata_regex, relation, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        relation_codes = match.group()
    return relation_codes


if __name__ == '__main__':
    dataset_list = [lcquad, qald9, qald, simple]


    for dataset in dataset_list:
            f = open(dataset, encoding='utf-8')
            data = json.load(f)
                # counter = 0
            if dataset != qald:
                for quest_set in data:
                    try:
                        relations = quest_set['relations_wikidata']
                        new_relations = [get_relation_code(item) for item in relations]
                        quest_set['relations_wikidata'] = new_relations

                    except json.decoder.JSONDecodeError:
                        pass
            elif dataset == qald:
                for quest_set in data:
                    try:
                        relations = quest_set['relations_wikidata']
                        true_predicates = quest_set['true_predicates']
                        new_relations = [get_relation_code(item) for item in relations]
                        new_true_pred = [get_relation_code(item) for item in true_predicates]
                        quest_set['relations_wikidata'] = new_relations
                        quest_set['true_predicates'] = new_true_pred

                    except json.decoder.JSONDecodeError:
                        pass

            if dataset == lcquad:
                setname = 'lcquad'
            elif dataset == simple:
                setname = 'simple_questions'
            elif dataset == qald:
                setname = 'qald'
            else:
                setname = 'qald_9'
            filename = f'../{setname}_results_corrected_relations.json'

            with open(filename, 'w', encoding='utf-8') as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)
