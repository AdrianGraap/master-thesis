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

    with open('../results/simplequestions_dbpedia_earl_results_NEW_MODEL.json', encoding='utf-8') as f:
        earl = json.load(f)

    # with open('../results/simple_questions_falcon_results.json', encoding='utf-8') as f:
    #     falcon = json.load(f)

    for earl_ques in earl:
        earl_ques['word_count'] = len([i for i in nltk.word_tokenize(earl_ques['question']) if i not in
                                       punctuation_list])

    # for falcon_ques in falcon:
    #     falcon_ques['word_count'] = len([i for i in nltk.word_tokenize(falcon_ques['question']) if i not in
    #                                      punctuation_list])

    with open('../results/simplequestions_dbpedia_earl_results_NEW_MODEL_new.json.json', 'w', encoding='utf-8') as outfile:
        json.dump(earl, outfile, ensure_ascii=False, indent=4)

    # with open('../results/simple_questions_falcon_results_new.json', 'w', encoding='utf-8') as outfile:
    #     json.dump(falcon, outfile, ensure_ascii=False, indent=4)
