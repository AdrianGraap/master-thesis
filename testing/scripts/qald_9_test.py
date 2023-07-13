import json
import re
import requests
import pickle
import pandas as pd

qald_9_wiki = '../datasets/qald_9_plus_test_wikidata.json'
qald_9_db = '../datasets/qald_9_plus_test_dbpedia.json'


def get_wikidata_relation(query):
    relations = []
    # wikidata_regex = r"<http:\/\/www.wikidata.org\/prop\/direct\/[a-zA-Z0-9]*>"
    wikidata_regex = r'P[0-9]+'
    matches = re.finditer(wikidata_regex, query, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        relations.append(match.group())
    return relations


def get_relation_code(relation):
    relation_codes = []
    wikidata_regex = r'P[0-9]+'
    matches = re.finditer(wikidata_regex, relation, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        return match.group()


def get_wikidata_entities(query):
    entities = []
    wikidata_regex = r"<http:\/\/www.wikidata.org\/entity\/[a-zA-Z0-9]*>"
    matches = re.finditer(wikidata_regex, query, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        entities.append(match.group())
    return entities


def get_dbpedia_relation(query):
    relations = []
    wikidata_regex = r"<http:\/\/www.wikidata.org\/prop\/direct\/[a-zA-Z0-9]*>"
    matches = re.finditer(wikidata_regex, query, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        relations.append(match.group())
    return relations


def get_dbpedia_entities(query):
    entities = []
    wikidata_regex = r"<http:\/\/www.wikidata.org\/entity\/[a-zA-Z0-9]*>"
    matches = re.finditer(wikidata_regex, query, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        entities.append(match.group())
    return entities


def get_earl(question: str):
    try:
        relation_list = []
        entity_list = []
        q = question.replace('"', '\\\"')
        # q.encode('utf-8')
        post_data = {"nlquery": q}
        post_data = json.dumps(post_data, ensure_ascii=True)
        post_data = post_data.encode('utf-8')
        headers = {'Content-type': 'application/json'}
        r = requests.post('http://localhost:4999/processQuery',
                          data=post_data, headers=headers)
        # json_acceptable_string = r.text.replace("'", "\"")
        answer = json.loads(r.text)

        for count, value in enumerate(answer['ertypes']):
            if value == 'relation':
                relation_list.append(answer['rerankedlists'][str(count)])
            elif value == 'entity':
                entity_list.append(answer['rerankedlists'][str(count)])
        return entity_list, relation_list
    except json.decoder.JSONDecodeError:
        raise


def get_falcon(question):
    # wikidata_regex = r"<http:\/\/www.wikidata.org\/entity\/([a-zA-Z0-9]*)>"
    try:
        q = question.replace('"', '\\\"')
        # q.encode('utf-8')
        post_data = {"text": q}
        post_data = json.dumps(post_data, ensure_ascii=True)
        post_data = post_data.encode('utf-8')
        headers = {'Content-type': 'application/json'}
        r = requests.post('https://labs.tib.eu/falcon/falcon2/api?mode=long&db=1',
                          data=post_data, headers=headers)
        # json_acceptable_string = r.text.replace("'", "\"")
        answer = json.loads(r.text)

        relations_dbpedia = [item[0] for item in answer['relations_dbpedia']]
        relations_wikidata = [get_relation_code(item[1]) for item in answer['relations_wikidata']]
        entities_dbpedia = [item[0] for item in answer['entities_dbpedia']]
        entities_wikidata = [item[1] for item in answer['entities_wikidata']]
    except:
        return [], [], [], []
    return relations_wikidata, relations_dbpedia, entities_wikidata, entities_dbpedia


def check_question(question, qa_system):
    if qa_system == 'earl':
        try:
            entity_list, relation_list = get_earl(question)
            return_dict = {'question': question,
                           'relations': relation_list,
                           'entities': entity_list}
        except json.decoder.JSONDecodeError:
            raise
    elif qa_system == 'falcon':
        relations_wikidata, relations_dbpedia, entities_wikidata, entities_dbpedia = get_falcon(question)
        return_dict = {'question': question,
                       'relations_wikidata': relations_wikidata,
                       'relations_dbpedia': relations_dbpedia,
                       'entities_wikidata': entities_wikidata,
                       'entities_dbpedia': entities_dbpedia}
    return return_dict


if __name__ == '__main__':
    correct = 0
    actual = 0

    # dataset = qald
    # system = 'earl'  # falcon OR earl
    system_list = ['earl']

    regex = r"(?<=<).*?(?=>)"
    # regex = r"(?<=(\\u003c)).*?(?=(\\u003e))"

    f = open(qald_9_wiki, encoding='utf-8')
    data_wiki = json.load(f)

    f = open(qald_9_db, encoding='utf-8')
    data_db = json.load(f)

    for system in system_list:
        all_dicts = []
        for quest_set_wiki in data_wiki['questions']:
            quest_set_db = [dic for dic in data_db['questions'] if dic['id'] == quest_set_wiki['id']][0]
            try:
                quest_wiki = quest_set_wiki['question'][0]['string']
                quest_db = quest_set_db['question'][0]['string']
                print(quest_wiki)

                sparql_query_wiki = quest_set_wiki['query']['sparql']
                sparql_query_db = quest_set_db['query']['sparql']

                true_entities_wiki = get_wikidata_entities(sparql_query_wiki)
                true_predicates_wiki = get_wikidata_relation(sparql_query_wiki)

                true_entities_db = []
                true_predicates_db = quest_set_db["true_relations"]

                system_dict = check_question(quest_wiki, system)
                system_dict['sparql_query_wikidata'] = sparql_query_wiki
                system_dict['sparql_query_dbpedia'] = sparql_query_db
                system_dict['true_entities_wikidata'] = true_entities_wiki
                system_dict['true_predicates_wikidata'] = true_predicates_wiki
                system_dict['true_entities_dbpedia'] = true_entities_db
                system_dict['true_predicates_dbpedia'] = true_predicates_db
                all_dicts.append(system_dict)
            except json.decoder.JSONDecodeError:
                pass

        filename = f'qald9_{system}_results_NEW_MODEL.json'

        try:
            with open(filename, 'w', encoding='utf-8') as outfile:
                json.dump(all_dicts, outfile, ensure_ascii=False, indent=4)
        except:
            backup_file = f'{filename}.pickle'
            file = open(backup_file, 'w')
            pickle.dump(all_dicts, file)

