import json
from get_superclass import get_superclass

if __name__ == '__main__':
    with open('../datasets/LCQuad.json', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        superclasses = []
        for relation in item['true_relations']:
            superclasses.append(get_superclass(relation))
        item['superclasses'] = superclasses

    with open('../datasets/LCQuad_superclasses.json', 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)