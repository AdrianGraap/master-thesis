import BLINK_most_specific_type
import BLINK_get_all_dbpedia_information
import json

lcquad = '..\\datasets\\LCQuad.json'


def annotate_questions(question_list):

    for quest in question_list:
        question_string = quest['question']
        for entity in quest['true_entites']:
            try:
                label = BLINK_get_all_dbpedia_information.get_label(entity)
                entity_type = BLINK_most_specific_type.get_most_specific_type(entity)
                all_relations = BLINK_get_all_dbpedia_information.get_all_relations(entity)
                all_relations = [element.split('/')[-1] for element in all_relations]
                question_string = f"""{question_string} [ {label} | {entity_type} | {', '.join(all_relations)} ]"""
            except:
                pass
        quest['GenRL_question'] = question_string
    with open('../datasets/LCQuad_GenRL', 'w',
              encoding='utf-8') as json_file:
        json.dump(question_list, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    with open(lcquad, encoding='utf-8') as f:
        lcquad_data = json.load(f)

    annotate_questions(lcquad_data)
#     ner_model = NER.get_model()
#
#     for item in questions:
#         annotation = main_dense.annotate(ner_model, [item])
#         item_dict = {
#             "id": 1,
#             "label": "unknown",
#             "label_id": -1,
#             "context_left": annotation[0]['context_left'].lower(),
#             "mention": annotation[0]['mention'].lower(),
#             "context_right": annotation[0]['context_right'].lower(),
#         }
#         data_to_link.append(item_dict)
#
#     _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
#
#     for pred, item in zip(predictions,questions):
#         entity: str = pred[0]
#         entity = entity.replace(' ', '_')
#         entity_type = most_specific_type.get_most_specific_type(entity)
#         all_relations = get_all_relations.get_all_relations(entity)
#         all_relations = [element.split('/')[-1] for element in all_relations]
#         question_string = f"""{item} [ {pred[0]} | {entity_type} | {', '.join(all_relations)} ]"""
#         print(question_string)

# print(main_dense.run(args, None, *models, test_data=data_to_link))
# print(data_to_link)
