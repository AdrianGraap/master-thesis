import re
import json

def get_all_relations(query):
    new_relation_list = []

    regex = r"""
        	\s*(<\S+>|\?\S+|\S+:\S+|;)\s+(<\S+>|\?\S+|\S+:\S+|a)\s+(<\S+>|\?\S+|\S+:\S+|\S+@\S+)\s*
        	"""

    matches = re.finditer(regex, query, re.VERBOSE)

    for matchNum, match in enumerate(matches, start=1):

        for groupNum in range(0, len(match.groups())):
            groupNum = groupNum + 1

            if groupNum == 2:
                new_relation = match.group(groupNum)
                new_relation_list.append(new_relation)
                # print(new_query)

    prefix_mapping = map_prefixes(query)
    new_relation_list = replace_prefixes(new_relation_list, prefix_mapping)

    return new_relation_list

def map_prefixes(query):
    mapping = {}

    regex = r"""
            	PREFIX\s+([a-zA-Z0-9]+):\s+<(\S+)>
            	"""

    matches = re.finditer(regex, query, re.VERBOSE)

    for matchNum, match in enumerate(matches, start=1):
        mapping[match.group(1)] = match.group(2)

    return mapping

def replace_prefixes(rel_list, mapping):
    return_rel_list = []
    for relation in rel_list:
        for key in mapping:
            replacement_string = f"{key}:"
            relation = relation.replace(replacement_string, mapping[key])
        relation = relation.replace('<', '').replace('>', '')
        if relation == 'a':
            relation = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
        return_rel_list.append(relation)

    return return_rel_list



if __name__ == '__main__':
    with open('../datasets/qald_9_plus_test_dbpedia.json', encoding='utf-8') as f:
        data = json.load(f)

    for item in data['questions']:
        relation_list = get_all_relations(item['query']['sparql'])
        if not relation_list:
            print(relation_list)
        item['true_relations'] = relation_list

    with open('../datasets/qald_9_plus_test_dbpedia.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)