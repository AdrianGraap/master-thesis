import pandas as pd
import os
# from matplotlib import pyplot as plt
import numpy as np
from rl_evaluator_changed import evaluate_using_dicts, load_gold_stardard, load_system_answers, evaluate_dbpedia_returning_dataframe
from matplotlib import pyplot as plt
import seaborn as sns
from mapping import file_mapping

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


# def create_bar_plot():
#     x = np.arange(avg.shape[0])
#     fig = plt.figure(figsize=(10, 8))
#     ax1 = fig.add_subplot(111)
#     ax2 = ax1.twiny()
#     ax1.bar(x - width, avg['Precision'], width, color='cyan')
#     ax1.bar(x, avg['Recall'], width, color='orange')
#     ax1.bar(x + width, avg['F1-Score'], width, color='green')
#     ax1.set_xticks(x, avg.index)
#     ax1.set_xlabel(r"Fragetypen")
#     ax1.legend(['Precision', 'Recall', 'F1-Score'])
#     ax1.set_title(f'Performance von {root} für verschiedene Fragetypen')
#     ax1.set_ylim(0, 1)
#     ax2.set_xlim(ax1.get_xlim())
#     ax2.set_xticks(x, count_df)
#     ax2.set_xticklabels(count_df)
#     ax2.set_xlabel(r"Anzahl Fragetypen")
#     plt.savefig(f'{root}/bar_plot_question_type.png')
#     plt.close()

#liste, welche ordner ausgelassn werden sollen
skip_list = ['LCQuad']

for root, subdirs, files in os.walk(directory):
    if 'eval' in root:
        if any(x in root for x in skip_list):
            continue
        if 'gold.json' in files and 'test.json' in files:
            path_gold = f'{root}\\gold.json'
            path_test = f'{root}\\test.json'

            gold_answers = load_gold_stardard(path_gold)
            system_answers = load_system_answers(path_test)

            df = evaluate_dbpedia_returning_dataframe(gold_answers, system_answers)
            print(df)

            corr = df.corr()

            corr = corr.drop(['precision', 'recall', 'f1', 'sys_relations'])
            x_axis_labels = ['Precision', 'Recall', 'F1-Score']

            corr = corr.drop(['rel_count', 'word_count', 'sys_relations'], axis=1)
            y_axis_labels = ['Wortanzahl', 'Anzahl Relationen']

            plt.figure(figsize=(12, 8))
            heat_map = sns.heatmap(corr, cmap="RdYlGn", annot=True, vmin=-1.0, vmax=1.0, xticklabels=x_axis_labels,
                                   yticklabels=y_axis_labels)
            heat_map.set(title=f"Korrelationsmatrix von {file_mapping[root]}")
            fig = heat_map.get_figure()
            fig.savefig(f'{root}/correlation.png')
