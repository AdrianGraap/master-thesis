import json
from candidate_list.my_own_candidate_list_scripts.entity_recognition import tag_entity, tag_entity_spacy, tag_entity_nltk
from candidate_list.my_own_candidate_list_scripts.get_candidates_from_kb import get_all_candidates
from tqdm import tqdm


def get_candidates_single_question(text, kb, toolkit='stanza'):
    entity_dict = dict()
    if toolkit == 'stanza':
        entities = tag_entity(text)
    elif toolkit == 'spacy':
        entities = tag_entity_spacy(text)
    elif toolkit == 'nltk':
        entities = tag_entity_nltk(text)
    else:
        raise Exception('toolkit muss stanza, spacy oder nltk sein')
    for entity in entities:
        entity_dict[entity] = get_all_candidates(entity, kb)
    return entity_dict


def main():
    with open('../../Wikidata/SMART2022-RL-wikidata-train.json', encoding='utf-8') as f:
        wikidata = json.load(f)
    with open('../../DBpedia/SMART2022-RL-dbpedia-train.json', encoding='utf-8') as f:
        dbpedia = json.load(f)

    for item in tqdm(wikidata):
        entity_list = list()
        entities = tag_entity_spacy(item['question'])
        for entity in entities:
            ent_list = [entity]
            ent_list.extend(get_all_candidates(entity, 'wikidata'))
            entity_list.append(ent_list)
        item['candidates'] = entity_list

    with open('SMART2022-RL-wikidata-train.json', 'w', encoding='utf-8') as outfile:
        json.dump(wikidata, outfile, ensure_ascii=False, indent=4)

    for item in tqdm(dbpedia):
        entity_list = list()
        entities = tag_entity_spacy(item['question'])
        for entity in entities:
            ent_list = [entity]
            ent_list.extend(get_all_candidates(entity, 'dbpedia'))
            entity_list.append(ent_list)
        item['candidates'] = entity_list

    with open('SMART2022-RL-dbpedia-train.json', 'w', encoding='utf-8') as outfile:
        json.dump(dbpedia, outfile, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
