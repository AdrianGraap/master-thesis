import json
import pandas as pd

if __name__ == '__main__':
    working_relation = set()
    not_working_relation = set()

    with open('../results_zwischenergebnisse/property_check.json', encoding='utf-8') as f:
        data = json.load(f)

    for quest in data:
        checklist = quest['relation_check']
        for index, item in enumerate(checklist):
            if (index % 2) == 0:
                saved_item = item
            else:
                if item:
                    working_relation.add(saved_item)
                else:
                    not_working_relation.add(saved_item)

    print(working_relation)
    print(not_working_relation)
    print('intersection', working_relation.intersection(not_working_relation))

    wrk = list(working_relation)
    not_wrk = list(not_working_relation)
    intr = list(working_relation.intersection(not_working_relation))
    dict = {'working': pd.Series(wrk), 'not_working': pd.Series(not_wrk), 'intersection': pd.Series(intr)}
    df = pd.DataFrame(dict)
    df.to_csv('../results/dbpedia_relations.csv')
