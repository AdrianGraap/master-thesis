import pandas as pd
import os
# from matplotlib import pyplot as plt
import numpy as np
from rl_evaluator_changed import evaluate_using_dicts, load_gold_stardard, load_system_answers, evaluate_dbpedia_returning_dataframe
from matplotlib import pyplot as plt
import seaborn as sns
# from mapping import file_mapping

directory = '..'
width = 0.2

def create_bar_plot():
    x = np.arange(avg.shape[0])
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.bar(x - width, avg['precision'], width, color='cyan')
    ax1.bar(x, avg['recall'], width, color='orange')
    ax1.bar(x + width, avg['f1'], width, color='green')
    for bars in ax1.containers:
        ax1.bar_label(bars, fmt='%.4f')
    ax1.set_xticks(x, avg.index)
    ax1.set_xlabel(r"Anzahl Relationen", fontsize=18)
    ax1.legend(['Precision', 'Recall', 'F1-Score'], fontsize=18)
    # ax1.set_title(f'Performance von {file_mapping[root]} nach Anzahlen der Relationen', wrap=True)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(x, count_df)
    ax2.set_xticklabels(count_df)
    ax2.set_xlabel(r"Anzahl Fragen", fontsize=18)
    ax1.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)
    plt.savefig(f'{root}/bar_plot_REL_COUNT.png')
    plt.close()


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

            # df = df.drop(df[df['sys_relations'] == 0].index)

            avg = df.groupby('rel_count').mean()
            count_df = df.groupby('rel_count')['question'].count()

            avg = avg[['precision', 'recall', 'f1']]

            create_bar_plot()
