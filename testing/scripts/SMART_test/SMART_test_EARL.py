import json
import requests


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

if __name__ == '__main__':
    with open('../../datasets/SMART2022-RL-dbpedia-train.json', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        item['predicted_entities'], item['predicted_relations'] = get_earl(item['question'])

    with open('../../results/SMART2022_EARL.json', 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)