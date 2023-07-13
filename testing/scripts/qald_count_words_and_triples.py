import json
import re
import nltk

def get_triple_count(query):
    # regex = """
    # \s*(<\S+>|\?\S+|\S+:\S+|;)\s+(<\S+>|\?\S+|\S+:\S+|a)\s+(<\S+>|\?\S+|\S+:\S+|\S+@\S+)\s*
    # """
    regex = """
    \s*(<\S+>|\?\S+|\S+:\S+|;)\s+(<\S+>|\?\S+|\S+:\S+|a|(<\S+>|\?\S+|\S+:\S+|a)\s*/\s*(<\S+>|\?\S+|\S+:\S+|a))\s+(<\S+>|\?\S+|\S+:\S+|\S+@\S+)\s*
    """

    return len([*re.finditer(regex, query, re.VERBOSE)])

if __name__ == '__main__':
    punctuation_list = [',', '.', '?']

    with open('../results/qald9_earl_results_NEW_MODEL.json', encoding='utf-8') as f:
        earl = json.load(f)

    # with open('../results/QALD9_FALCON.json', encoding='utf-8') as f:
    #     falcon = json.load(f)

    for earl_ques in earl:
        earl_ques['word_count'] = len([i for i in nltk.word_tokenize(earl_ques['question'])
                                                                if i not in punctuation_list])

        earl_ques['triple_count_dbpedia'] = get_triple_count(earl_ques['sparql_query_dbpedia'])

        earl_ques['triple_count_wikidata'] = get_triple_count(earl_ques['sparql_query_wikidata'])

    # for falcon_ques in falcon:
    #     falcon_ques['word_count'] = len([i for i in nltk.word_tokenize(falcon_ques['question'])
    #                                                             if i not in punctuation_list])
    #
    #     falcon_ques['triple_count_dbpedia'] = get_triple_count(falcon_ques['sparql_query_dbpedia'])
    #
    #     falcon_ques['triple_count_wikidata'] = get_triple_count(falcon_ques['sparql_query_wikidata'])

    with open('../results/qald9_earl_results_NEW_MODEL_new.json', 'w', encoding='utf-8') as outfile:
        json.dump(earl, outfile, ensure_ascii=False, indent=4)

    # with open('../results/QALD9_FALCON_new.json', 'w', encoding='utf-8') as outfile:
    #     json.dump(falcon, outfile, ensure_ascii=False, indent=4)
