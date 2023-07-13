import stanza
import spacy
import nltk
import json

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

nlp = stanza.Pipeline('en', processors='tokenize,ner')
nlp_spacy = spacy.load('en_core_web_sm')

entity_exclusion = ['DATE', 'CARDINAL']


def tag_entity(text):
    doc = nlp(text)
    # print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
    return [ent.text for ent in doc.ents if ent.type not in entity_exclusion]


def tag_entity_spacy(text):
    doc = nlp_spacy(text)
    return [ent.text for ent in doc.ents if ent.label_ not in entity_exclusion]


def tag_entity_nltk(text):
    for sent in nltk.sent_tokenize(text):
        return [' '.join(c[0] for c in chunk) for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)))
                if (hasattr(chunk, 'label')) and (chunk.label() not in entity_exclusion)]


def main():
    with open('../../Wikidata/SMART2022-RL-wikidata-train.json', encoding='utf-8') as f:
        wikidata = json.load(f)
    for item in wikidata:
        # sentence = item['question']
        sentence = 'When did Arvo Pärt receive an honorary Doctorate from the University of Liège?'
        print(sentence)
        print(tag_entity(sentence))
        print(tag_entity_spacy(sentence))
        print(tag_entity_nltk(sentence))
        print('---------------------------')


if __name__ == '__main__':
    main()
