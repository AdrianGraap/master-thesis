import json
import re
import nltk

def get_triple_count(query):
    regex = """
    \s*(<\S+>|\?\S+|\S+:\S+|;)\s+(<\S+>|\?\S+|\S+:\S+|a)\s+(<\S+>|\?\S+|\S+:\S+|\S+@\S+)\s*
    """

    return len([*re.finditer(regex, query, re.VERBOSE)])

if __name__ == '__main__':
    punctuation_list = [',', '.', '?']

    with open('../results/LCQuad_EARL_corrected.json', encoding='utf-8') as f:
        earl = json.load(f)

    with open('../results/LCQuad_FALCON_corrected.json', encoding='utf-8') as f:
        falcon = json.load(f)

    for earl_ques, falcon_quest in zip(earl, falcon):
        earl_ques['word_count'] = falcon_quest['word_count'] = len([i for i in nltk.word_tokenize(earl_ques['question'])
                                                                if i not in punctuation_list])

        earl_ques['triple_count'] = falcon_quest['triple_count'] = get_triple_count(earl_ques['sparql_query'])

    with open('../results/qald9_earl_results_CORRECTED_new.json', 'w', encoding='utf-8') as outfile:
        json.dump(earl, outfile, ensure_ascii=False, indent=4)

    with open('../results/LCQuad_FALCON_corrected_new.json', 'w', encoding='utf-8') as outfile:
        json.dump(falcon, outfile, ensure_ascii=False, indent=4)
