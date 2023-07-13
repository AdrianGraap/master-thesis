import json
from SPARQLWrapper import SPARQLWrapper, JSON
import os, ssl
import re
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

def replace_relation(query):
    new_relation_list = []
    new_entity_list = []

    regex = r"""
    	\s*(<\S+>|\?\S+)\s+(<\S+>|\?\S+)\s+(<\S+>|\?\S+)\s*
    	"""

    matches = re.finditer(regex, query, re.VERBOSE)

    for matchNum, match in enumerate(matches, start=1):


        for groupNum in range(0, len(match.groups())):
            groupNum = groupNum + 1

            if (groupNum == 1) or (groupNum == 3):
                new_entity = match.group(groupNum)
                new_entity = new_entity.replace('<', '').replace('>', '')
                if 'dbpedia' in new_entity:
                    new_entity_list.append(new_entity)

            if groupNum == 2:
                new_relation = match.group(groupNum)
                new_relation = new_relation.replace('<', '').replace('>', '')
                new_relation_list.append(new_relation)
                # print(new_query)
    return new_entity_list, new_relation_list

if __name__ == '__main__':
    with open('../datasets/FullyAnnotated_LCQuAD5000.json', encoding='utf-8') as file:
        data = json.load(file)

    for quest in data:
        original_query = quest['sparql_query']
        changed_entity_list, changed_relation_list = replace_relation(original_query)
        quest['true_relations'] = changed_relation_list
        quest['true_entities'] = changed_entity_list

    with open('../datasets/LCQuad_new.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
