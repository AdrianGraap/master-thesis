import json
from SPARQLWrapper import SPARQLWrapper, JSON
import os, ssl
import re
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

def replace_relation(query):
    new_query_list = []
    new_relation_list = []

    regex = r"""
    	\s*(<\S+>|\?\S+)\s+(<\S+>|\?\S+)\s+(<\S+>|\?\S+)\s*
    	"""

    matches = re.finditer(regex, query, re.VERBOSE)

    for matchNum, match in enumerate(matches, start=1):


        for groupNum in range(0, len(match.groups())):
            groupNum = groupNum + 1

            if groupNum == 2:
                new_relation = match.group(groupNum).replace('ontology', 'property')
                # print(match.group(groupNum), new_relation)
                new_query = query[0:match.start(groupNum)] + new_relation + query[match.end(groupNum):len(query)]
                if 'property' in new_relation:
                    new_query_list.append(new_query)
                    new_relation_list.append(new_relation)
                # print(new_query)
    return new_query_list, new_relation_list

if __name__ == '__main__':


    with open('../datasets/FullyAnnotated_LCQuAD5000.json', encoding='utf-8') as file:
        data = json.load(file)


    for quest in data:
        relation_list = []
        original_query = quest['sparql_query']
        original_query = original_query.replace('https', 'http')
        dbpedia_endpoint = "https://dbpedia.org/sparql"
        sparql = SPARQLWrapper(dbpedia_endpoint)

        try:
            sparql.setQuery(original_query)
            sparql.setReturnFormat(JSON)
            original_resp = sparql.query().convert()  # answer from the SPARQL Endpoint of a KG
        except:
            original_resp = []
        # changed_query = original_query.replace('/ontology/', '/property/')
        changed_query_list, changed_relation_list = replace_relation(original_query)
        for changed_query, changed_relation in zip(changed_query_list, changed_relation_list):

            try:
                sparql.setQuery(changed_query)
                sparql.setReturnFormat(JSON)
                changed_resp = sparql.query().convert()  # answer from the SPARQL Endpoint of a KG
            except:
                changed_resp = []

            if 'results' in original_resp:
                if original_resp['results']['bindings']:
                    if original_resp['results']['bindings'] == changed_resp['results']['bindings']:
                        relation_list.append(changed_relation)
                        relation_list.append(True)
                        # quest['property_check'] = True
                    else:
                        relation_list.append(changed_relation)
                        relation_list.append(False)
        quest['relation_check'] = relation_list

    with open('../results_zwischenergebnisse/property_check.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)