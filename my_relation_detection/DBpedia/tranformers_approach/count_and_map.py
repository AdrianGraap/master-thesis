import json


def count_and_map(file='../SMART2022-RL-dbpedia-train.json'):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)

    all_relation_lists = dict()
    label2id = dict()

    for item in data:
        mapped_relations = []
        for rel in item['relations']:
            if tuple(rel) in all_relation_lists:
                mapped_relations.append(all_relation_lists[tuple(rel)])
            else:
                all_relation_lists[tuple(rel)] = len(all_relation_lists)
                mapped_relations.append(all_relation_lists[tuple(rel)])
        item['mapped_relations'] = mapped_relations
    for key, value in all_relation_lists.items():
        key = '|'.join(key)
        label2id[key] = value
    with open('label2id.json', 'w') as outfile:
        json.dump(label2id, outfile)
    return data, all_relation_lists


def count_and_map_superclasses(file='../SMART_superclasses.json'):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)

    all_relation_lists = dict()
    label2id = dict()

    for item in data:
        mapped_superclasses = []
        for rel in item['superclasses']:
            if rel in all_relation_lists:
                mapped_superclasses.append(all_relation_lists[rel])
            else:
                all_relation_lists[rel] = len(all_relation_lists)
                mapped_superclasses.append(all_relation_lists[rel])
        item['mapped_superclasses'] = mapped_superclasses
    for key, value in all_relation_lists.items():
        label2id[key] = value
    with open('label2id_superclasses.json', 'w') as outfile:
        json.dump(label2id, outfile)
    return data, all_relation_lists


def count_and_map_first_item(file='../SMART2022-RL-dbpedia-train.json'):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)

    all_relation_lists = dict()
    label2id = dict()

    for item in data:
        mapped_relations = []
        for rel in item['relations']:
            if rel[0] in all_relation_lists:
                mapped_relations.append(all_relation_lists[rel[0]])
            else:
                all_relation_lists[rel[0]] = len(all_relation_lists)
                mapped_relations.append(all_relation_lists[rel[0]])
        item['mapped_relations'] = mapped_relations
    for key, value in all_relation_lists.items():
        label2id[key] = value
    with open('label2id_first_item.json', 'w') as outfile:
        json.dump(label2id, outfile)
    return data, all_relation_lists


def count_and_map_all_items(num_labels, file='../SMART2022-RL-dbpedia-train.json'):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)

    with open('../SMART2022-RL-dbpedia-relation-vocabulary.json', encoding='utf-8') as f:
        vocab = json.load(f)

    # TODO: label2id muss auf ALLE Relationen erweutert werden und das in 3facher ausführung

    all_relation_lists = dict()
    label2id = dict()

    for counter in range(3):
        for i, v in enumerate(vocab):
            label2id[f'{v}|{counter}'] = num_labels * counter + i
        # label2id = {f'{k}|{number}': i for number in range(3) for k in vocab}

    for item in data:
        item['mapped_relations'] = []
        for i, rel_list in enumerate(item['relations']):
            mapped_relations = []
            for rel in rel_list:
                rel_for_mapping = f'{rel}|{i}'
                mapped_relations.append(label2id[rel_for_mapping])
            item['mapped_relations'].append(mapped_relations)
    # for key, value in all_relation_lists.items():
    #     label2id[key] = value
    with open('label2id_first_item.json', 'w') as outfile:
        json.dump(label2id, outfile)
    return data, label2id


def count_and_map_all_items_short_list(file='../SMART2022-RL-dbpedia-train.json', vocab_file='../SMART2022-RL-dbpedia-relation-vocabulary.json'):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)

    with open(vocab_file, encoding='utf-8') as f:
        vocab = json.load(f)

    label2id = {v: i for i, v in enumerate(vocab)}

    for item in data:
        mapped_relations = []
        for rel_list in item['relations']:
            for rel in rel_list:
                mapped_relations.append(label2id[rel])
        item['mapped_relations'] = mapped_relations
    with open('label2id.json', 'w') as outfile:
        json.dump(label2id, outfile)
    return data, label2id


def count_and_map_different_models(model_name, file='../SMART2022-RL-dbpedia-train.json'):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)

    all_relation_lists = dict()
    label2id = dict()

    for item in data:
        mapped_relations = []
        for rel in item['relations']:
            if rel[0] in all_relation_lists:
                mapped_relations.append(all_relation_lists[rel[0]])
            else:
                all_relation_lists[rel[0]] = len(all_relation_lists)
                mapped_relations.append(all_relation_lists[rel[0]])
        item['mapped_relations'] = mapped_relations
    for key, value in all_relation_lists.items():
        label2id[key] = value
    with open(f'label2id_{model_name}.json', 'w') as outfile:
        json.dump(label2id, outfile)
    return data, all_relation_lists


def count_and_map_qald(file='../../testing/datasets/qald_9_plus_test_dbpedia.json'):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)

    all_relation_lists = dict()
    label2id = dict()
    rel_vocab = set()

    for item in data['questions']:
        mapped_relations = []
        for rel in item['true_relations']:
            rel_vocab.add(rel)
            if rel in all_relation_lists:
                mapped_relations.append(all_relation_lists[rel])
            else:
                all_relation_lists[rel] = len(all_relation_lists)
                mapped_relations.append(all_relation_lists[rel])
        item['mapped_relations'] = mapped_relations
    for key, value in all_relation_lists.items():
        label2id[key] = value
    with open('label2id_qald.json', 'w') as outfile:
        json.dump(label2id, outfile)
    return data, all_relation_lists, rel_vocab


def count_and_map_lcquad(file='../../testing/datasets/LCQuad.json'):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)

    all_relation_lists = dict()
    label2id = dict()
    rel_vocab = set()

    for item in data:
        mapped_relations = []
        for rel in item['true_relations']:
            rel_vocab.add(rel)
            if rel in all_relation_lists:
                mapped_relations.append(all_relation_lists[rel])
            else:
                all_relation_lists[rel] = len(all_relation_lists)
                mapped_relations.append(all_relation_lists[rel])
        item['mapped_relations'] = mapped_relations
    for key, value in all_relation_lists.items():
        label2id[key] = value
    with open('label2id_lcquad.json', 'w') as outfile:
        json.dump(label2id, outfile)
    return data, all_relation_lists, rel_vocab


if __name__ == '__main__':
    count_and_map()
