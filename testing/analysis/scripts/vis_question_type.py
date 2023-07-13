import pandas as pd
import os
from get_dataframe import get_dataframe
from get_question_type import check
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    directory = '../csv/'
    width = 0.2

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if ('lcquad' in filename) or ('qald' in filename) or ('SMART' in filename):
            full_file = os.path.join(directory, filename)
            df = get_dataframe(full_file)
            df.dropna(inplace=True)

            if ('lcquad' in filename) or ('qald' in filename):
                df['question_type'] = df.apply(lambda row: check(row['params.question'], row['params.query_db']), axis=1)
            elif 'SMART' in filename:
                df.rename(columns={'params.question_type': 'question_type'}, inplace=True)
            avg = df.groupby('question_type').mean()
            count_df = df.groupby('question_type')['params.question'].count()
            avg = avg[['metrics.precision', 'metrics.recall', 'metrics.f1']]
            x = np.arange(avg.shape[0])
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()
            ax1.bar(x-width, avg['metrics.precision'], width, color='cyan')
            ax1.bar(x, avg['metrics.recall'], width, color='orange')
            ax1.bar(x + width, avg['metrics.f1'], width, color='green')
            ax1.set_xticks(x, avg.index)
            ax1.set_xlabel(r"Fragetypen", fontsize=18)
            ax1.legend(['Precision', 'Recall', 'F1-Score'], fontsize=18)
            if 'lcquad' in filename:
                dataset = 'LC-QuAD 1.0'
            elif 'qald' in filename:
                dataset = 'QALD 9'
            elif 'simple' in filename:
                dataset = 'SimpleQuestions'
            elif 'SMART' in filename:
                dataset = 'SMART'
            # ax1.set_title(f'Performance von {filename.split(".")[0].split("_")[0]} auf {dataset} für verschiedene Fragetypen')
            ax1.set_ylim(0,1)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(x, count_df)
            ax2.set_xticklabels(count_df)
            ax2.set_xlabel(r"Anzahl Fragetypen", fontsize=18)
            ax1.tick_params(labelsize=18)
            ax2.tick_params(labelsize=18)
            plt.savefig(f'../bar plots/{filename.split(".")[0].split("_")[0]}_{dataset}_bar_plot_question_type.png')
            plt.close()
