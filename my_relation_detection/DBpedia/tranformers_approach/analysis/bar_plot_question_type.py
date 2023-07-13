import pandas as pd
import os
# from matplotlib import pyplot as plt
import numpy as np
from rl_evaluator_changed import evaluate_using_dicts, load_gold_stardard, load_system_answers, evaluate_dbpedia_returning_dataframe
from matplotlib import pyplot as plt
# from mapping import file_mapping

directory = '..'
width = 0.2

bool_gold_dict = dict()
count_gold_dict = dict()
factoid_gold_dict = dict()
list_gold_dict = dict()
bool_system_dict = dict()
count_system_dict = dict()
factoid_system_dict = dict()
list_system_dict = dict()


def create_bar_plot():
    x = np.arange(avg.shape[0])
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.bar(x - width, avg['precision'], width, color='cyan')
    ax1.bar(x, avg['recall'], width, color='orange')
    ax1.bar(x + width, avg['f1'], width, color='green')
    for bars in ax1.containers:
        ax1.bar_label(bars, fmt='%.3f')
    ax1.set_xticks(x, avg.index)
    ax1.set_xlabel(r"Fragetypen", fontsize=18)
    ax1.legend(['Precision', 'Recall', 'F1-Score'], fontsize=18)
    # ax1.set_title(f'Performance von {file_mapping[root]} für verschiedene Fragetypen', wrap=True)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(x, count_df)
    ax2.set_xticklabels(count_df)
    ax2.set_xlabel(r"Anzahl Fragetypen", fontsize=18)
    ax1.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)
    plt.savefig(f'{root}/bar_plot_question_type.png')
    plt.close()

for root, subdirs, files in os.walk(directory):
    if 'eval' in root:
        if 'gold.json' in files and 'test.json' in files:
            path_gold = f'{root}\\gold.json'
            path_test = f'{root}\\test.json'

            gold_answers = load_gold_stardard(path_gold)
            system_answers = load_system_answers(path_test)

            df = evaluate_dbpedia_returning_dataframe(gold_answers, system_answers)

            avg = df.groupby('question_type').mean()
            count_df = df.groupby('question_type')['question'].count()

            avg = avg.drop('error', 0, errors='ignore')
            count_df = count_df.drop('error', 0, errors='ignore')

            avg = avg[['precision', 'recall', 'f1']]

            create_bar_plot()

            # count = np.arange(4)
            # error_count = 0
            #
            # for (gold_id, gold_dict), (system_id, system_dict) in zip(gold_answers.items(), system_answers.items()):
            #     # print(gold_id, gold_dict, system_id, system_dict)
            #     if gold_dict['question_type'] == 'boolean':
            #         bool_gold_dict[gold_id] = gold_dict['relations']
            #         bool_system_dict[gold_id] = system_dict
            #         count[0] += 1
            #     elif gold_dict['question_type'] == 'count':
            #         count_gold_dict[gold_id] = gold_dict['relations']
            #         count_system_dict[gold_id] = system_dict
            #         count[1] += 1
            #     elif gold_dict['question_type'] == 'factoid':
            #         factoid_gold_dict[gold_id] = gold_dict['relations']
            #         factoid_system_dict[gold_id] = system_dict
            #         count[2] += 1
            #     elif gold_dict['question_type'] == 'list':
            #         list_gold_dict[gold_id] = gold_dict['relations']
            #         list_system_dict[gold_id] = system_dict
            #         count[3] += 1
            #     else:
            #         error_count += 1
            #
            # p1, r1, f1_1 = evaluate_using_dicts(bool_gold_dict, bool_system_dict, 'dbpedia')
            # p2, r2, f1_2 = evaluate_using_dicts(count_gold_dict, count_system_dict, 'dbpedia')
            # p3, r3, f1_3 = evaluate_using_dicts(factoid_gold_dict, factoid_system_dict, 'dbpedia')
            # p4, r4, f1_4 = evaluate_using_dicts(list_gold_dict, list_system_dict, 'dbpedia')
            #
            # df_dict = {
            #     'Precision': [p1, p2, p3, p4],
            #     'Recall': [r1, r2, r3, r4],
            #     'F1-Score': [f1_1, f1_2, f1_3, f1_4]
            # }
            # df_index = ['boolean', 'count', 'factoid', 'list']
            # avg = pd.DataFrame(df_dict, index=df_index)
            # count_df = pd.Series(count, index=df_index)
            #
            # print(root, error_count)

            # create_bar_plot()
