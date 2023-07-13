import pandas as pd
import json

lcquad_falcon = '../lcquad_results_corrected_relations.json'
qald_falcon = '../qald_results_corrected_relations.json'
simple_falcon = '../simple_questions_results_corrected_relations.json'
qald_9 = '../qald_9_results_corrected_relations.json'
qald_9_no_request = '../qald9_falcon_results_no_request.json'

datasets = [lcquad_falcon, qald_falcon, simple_falcon, qald_9]

if __name__ == '__main__':
    for filename in datasets:
        with open(filename, 'rb') as file:
            data = json.load(file)

        dataframe = pd.DataFrame(data)

        filename = filename.split('.')[-2][1:]

        dataframe.to_csv(f'../csv_results/{filename}.csv', sep='|')
