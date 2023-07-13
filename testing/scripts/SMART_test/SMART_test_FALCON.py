import json
import requests
import re


def get_relation_code(relation):
    relation_codes = []
    wikidata_regex = r'P[0-9]+'
    matches = re.finditer(wikidata_regex, relation, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        return match.group()


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

        relations_dbpedia = [item['URI'] for item in answer['relations_dbpedia']]
        relations_wikidata = [get_relation_code(item['URI']) for item in answer['relations_wikidata']]
        entities_dbpedia = [item['URI'] for item in answer['entities_dbpedia']]
        entities_wikidata = [item['URI'] for item in answer['entities_wikidata']]
    except:
        return [], [], [], []
    return relations_wikidata, relations_dbpedia, entities_wikidata, entities_dbpedia

if __name__ == '__main__':
    with open('../../datasets/SMART2022-RL-dbpedia-train.json', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        _, item['relations'], _, item['entities'] = get_falcon(item['question'])

    with open('../../results/SMART2022_FALCON_dbpedia.json', 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    with open('../../datasets/SMART2022-RL-wikidata-train.json', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        item['relations'], _, item['entities'], _ = get_falcon(item['question'])

    with open('../../results/SMART2022_FALCON_wikidata.json', 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)