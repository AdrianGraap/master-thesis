import json
import re
import requests
import pickle
import pandas as pd

lcquad = '../datasets/LCQuad.json'
qald = '../datasets/qald-7-test-en-wikidata.json'
simpleq = '../datasets/simple_questions.csv'
simpleq_db = '../datasets/simple_questions_dbpedia.json'
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
        relation_codes.append(match.group())
    return relation_codes


def get_wikidata_entities(query):
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
        # r = requests.post('http://localhost:4999/processQuery',
        #                   data=post_data, headers=headers)
        r = requests.post('http://ltdemos.informatik.uni-hamburg.de/earl/processQuery',
                          data=post_data, headers=headers)
        # json_acceptable_string = r.text.replace("'", "\"")
        answer = json.loads(r.text)

        for count, value in enumerate(answer['ertypes']):
            if value == 'relation':
                relation_list.append(answer['rerankedlists'][str(count)])
            elif value == 'entity':
                entity_list.append(answer['rerankedlists'][str(count)])
        return entity_list, relation_list
    except json.decoder.JSONDecodeError as e:
        print(str(e))
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

    dataset_list = [lcquad]
    system_list = ['earl']

    # dataset = qald
    # system = 'earl'  # falcon OR earl

    regex = r"(?<=<).*?(?=>)"
    # regex = r"(?<=(\\u003c)).*?(?=(\\u003e))"

    for dataset in dataset_list:
        try:
            for system in system_list:
                all_dicts = []
        # ---LC-QUAD
                if dataset == lcquad:
                    f = open(dataset, encoding='utf-8')
                    data = json.load(f)
                    # counter = 0
                    for quest_set in data:
                        try:
                            quest = quest_set['question']
                            print(quest)
                            sparql_query = quest_set['sparql_query']
                            # true_entities = [item['uri'] for item in quest_set['entity mapping']]     #GEÄNDERT FÜR KORRIGIERTEN DATENSATZ
                            # true_predicates = [item['uri'] for item in quest_set['predicate mapping']]
                            # matches = re.finditer(regex, quest_set['sparql_query'], re.MULTILINE)
                            # for matchNum, match in enumerate(matches, start=1):
                            #     all_tokens.append(match.group())

                            system_dict = check_question(quest, system)
                            system_dict['sparql_query'] = sparql_query
                            system_dict['true_entities'] = quest_set['true_entities']
                            system_dict['true_predicates'] = quest_set['true_relations']
                            all_dicts.append(system_dict)
                            # counter += 1
                            # if counter == 10:
                            #     break
                        except json.decoder.JSONDecodeError:
                            pass

                elif dataset == qald:
                    f = open(dataset, encoding='utf-8')
                    data = json.load(f)
                    for quest_set in data['questions']:
                        try:
                            quest = quest_set['question'][0]['string']
                            print(quest)

                            sparql_query = quest_set['query']['sparql']
                            true_entities = get_wikidata_entities(sparql_query)
                            true_predicates = get_wikidata_relation(sparql_query)

                            system_dict = check_question(quest, system)
                            system_dict['sparql_query'] = sparql_query
                            system_dict['true_entities'] = true_entities
                            system_dict['true_predicates'] = true_predicates
                            all_dicts.append(system_dict)
                        except json.decoder.JSONDecodeError:
                            pass
                elif dataset == simpleq:
                    try:
                        simple_data = pd.read_csv(simpleq, delimiter=',', index_col=False,
                                                  converters={'Gold Standard Entities': lambda x: x.strip("[]").strip("'").split(", "),
                                                              'Gold Standard Relations': lambda x: x.strip("[]").strip("'").split(", ")})
                        for index, row in simple_data.iterrows():
                            quest = row['Question']
                            print(quest)
                            # sparql_query = quest_set['sparql_query']
                            true_entities = row['Gold Standard Entities']
                            true_predicates = row['Gold Standard Relations']

                            system_dict = check_question(quest, system)
                            system_dict['true_entities'] = true_entities
                            system_dict['true_predicates'] = true_predicates
                            all_dicts.append(system_dict)
                            # counter += 1
                            # if counter == 10:
                            #     break

                    except json.decoder.JSONDecodeError:
                        pass
                elif dataset == simpleq_db:
                    f = open(dataset, encoding='utf-8')
                    data = json.load(f)

                    for quest_set in data['Questions']:
                        try:
                            quest = quest_set['Query']

                            true_entities = quest_set['Subject']
                            true_predicates = [pred['Predicate'] for pred in quest_set['PredicateList']]

                            system_dict = check_question(quest, system)
                            system_dict['sparql_query'] = ''
                            system_dict['true_entities'] = true_entities
                            system_dict['true_predicates'] = true_predicates
                            all_dicts.append(system_dict)
                        except json.decoder.JSONDecodeError:
                            pass

                if dataset == lcquad:
                    setname = 'lcquad'
                elif dataset == simpleq:
                    setname = 'simple_questions'
                elif dataset == simpleq_db:
                    setname = 'simplequestions_dbpedia'
                else:
                    setname = 'qald'
                filename = f'{setname}_{system}_results_SERVICE.json'

                try:
                    with open(filename, 'w', encoding='utf-8') as outfile:
                        json.dump(all_dicts, outfile, ensure_ascii=False, indent=4)
                except:
                    backup_file = f'{filename}.pickle'
                    file = open(backup_file, 'w')
                    pickle.dump(all_dicts, file)
        except KeyboardInterrupt:
            setname = 'ERROR'
            if dataset == lcquad:
                setname = 'lcquad'
            elif dataset == simpleq:
                setname = 'simple_questions'
            elif dataset == simpleq_db:
                setname = 'simplequestions_dbpedia'
            else:
                setname = 'qald'
            filename = f'{setname}_{system}_results_SERVICE.json'
            with open(filename, 'w', encoding='utf-8') as outfile:
                json.dump(all_dicts, outfile, ensure_ascii=False, indent=4)

