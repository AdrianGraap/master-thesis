import json

if __name__ == '__main__':

    with open('../qald_earl_results.json', 'rb') as f:
        data = json.load(f)

    for item in data:
        relations = []
        for relation in item['relations']:
            relations.append(relation[0][1])
        entities = []
        for entity in item['entities']:
            entities.append(entity[0][1])
