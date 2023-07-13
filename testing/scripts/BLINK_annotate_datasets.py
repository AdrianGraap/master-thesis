import json
import BLINK_annotate_questions

lcquad = 'datasets\\LCQuad.json'
qald9_dbpedia = 'datasets\\qald_9_plus_test_dbpedia.json'
qald9_wiki = 'datasets\\qald_9_plus_test_wikidata.json'
simple = 'datasets\\simple_questions_dbpedia.json'

if __name__ == '__main__':
    with open(lcquad, encoding='utf-8') as f:
        lcquad_data = json.load(f)

    with open(qald9_dbpedia, encoding='utf-8') as f:
        qald9_dbpedia_data = json.load(f)

    with open(qald9_wiki, encoding='utf-8') as f:
        qald9_wiki_data = json.load(f)

    with open(simple, encoding='utf-8') as f:
        simple_data = json.load(f)

        # LCQUAD

    item_list = []

    for item in lcquad_data:
        item_list.append(item['question'])

    annotated_list = annotate_questions.annotate_questions(item_list)

    for item, annotated_question in zip(lcquad_data, annotated_list):
        item['question_GenRL'] = annotated_question

    with open('datasets.LCQuad_GenRL', 'w',
              encoding='utf-8') as json_file:
        json.dump(lcquad_data, json_file, ensure_ascii=False, indent=4)

    #     # SIMPLE
    #
    # item_list = []
    #
    # for item in simple_data['Questions']:
    #     item_list.append(item['Query'])
    #
    # annotated_list = annotate_questions.annotate_questions(item_list)
    #
    # for item, annotated_question in zip(simple_data['Questions'], annotated_list):
    #     item['question_GenRL'] = annotated_question
    #
    # with open('E:\\Hochschule\\Master\\Masterarbeit\\testing\\datasets.Simple_GenRL', 'w',
    #           encoding='utf-8') as json_file:
    #     json.dump(lcquad_data, json_file, ensure_ascii=False, indent=4)
    #
    #     # QALD
    #
    # item_list = []
    #
    # for item in qald9_dbpedia_data['questions']:
    #     item_list.append(item['question'][0]['string'])
    #
    # annotated_list = annotate_questions.annotate_questions(item_list)
    #
    # for item_db, item_wiki, annotated_question in zip(qald9_dbpedia_data['questions'], qald9_wiki_data['questions'], annotated_list):
    #     item_db['question_GenRL'] = annotated_question
    #     item_wiki['question_GenRL'] = annotated_question
    #
    # with open('E:\\Hochschule\\Master\\Masterarbeit\\testing\\datasets.QALD_DB_GenRL', 'w', encoding='utf-8') as json_file:
    #     json.dump(qald9_dbpedia_data, json_file, ensure_ascii=False, indent=4)
    #
    # with open('E:\\Hochschule\\Master\\Masterarbeit\\testing\\datasets.QALD_WIKI_GenRL', 'w', encoding='utf-8') as json_file:
    #     json.dump(qald9_wiki_data, json_file, ensure_ascii=False, indent=4)