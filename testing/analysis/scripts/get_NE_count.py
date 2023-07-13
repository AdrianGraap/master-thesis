import re


def get_ne_count(query):
    ne_count = 0
    entity_list = []

    regex = r"""\s*(<\S+>|\?\S+|\S+:\S+|;)\s+(<\S+>|\?\S+|\S+:\S+|a)\s+(<\S+>|\?\S+|\S+:\S+|\S+@\S+)\s*"""

    matches = re.finditer(regex, query, re.VERBOSE)

    for matchNum, match in enumerate(matches, start=1):

        for groupNum in range(0, len(match.groups())):
            groupNum = groupNum + 1

            if (groupNum == 1) or (groupNum == 3):
                new_entity = match.group(groupNum)
                # new_entity = new_entity.replace('<', '').replace('>', '')
                if ('?' not in new_entity) and (';' not in new_entity) and (new_entity not in entity_list):
                    ne_count += 1
                    entity_list.append(new_entity)

    return ne_count

if __name__ == '__main__':
    print(get_ne_count('PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX dct: <http://purl.org/dc/terms/> PREFIX dbc: <http://dbpedia.org/resource/Category:> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT ?uri WHERE { ?uri rdf:type dbo:Ship ; dct:subject dbc:Christopher_Columbus ; dct:subject dbc:Exploration_ships }'))
