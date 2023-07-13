import json

def main():
    count_dict = dict()

    with open('../SMART2022-RL-dbpedia-train.json', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        for sublist in item['relations']:
            for relation in sublist:
                if relation in count_dict:
                    count_dict[relation] += 1
                else:
                    count_dict[relation] = 1

    with open('relation_count.json', 'w') as outfile:
        json.dump(count_dict, outfile)

if __name__ == '__main__':
    main()