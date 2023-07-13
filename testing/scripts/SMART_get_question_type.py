import json
from question_type import check_quest

dbpedia = '../datasets/SMART2022-RL-dbpedia-train.json'
wikidata = '../datasets/SMART2022-RL-wikidata-train.json'


if __name__ == '__main__':
    with open(wikidata, encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        item['question_type'] = check_quest(item['question'])

    with open(wikidata, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)
