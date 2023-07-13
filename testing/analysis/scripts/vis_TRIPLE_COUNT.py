import pandas as pd
import os
from get_dataframe import get_dataframe
from get_NE_count import get_ne_count
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    directory = '../csv/'
    width = 0.2

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        full_file = os.path.join(directory, filename)
        df = get_dataframe(full_file)
        df.dropna(inplace=True)
        if 'SMART' in filename:
            continue

        if ('lcquad' in filename):
            column = 'metrics.triple_count'
        elif ('qald' in filename):
            column = 'metrics.triple_count_dbpedia'
        elif ('simple' in filename):
            df['triple_count'] = 1
            column = 'triple_count'

        if 'lcquad' in filename:
            dataset = 'LC-QuAD 1.0'
        elif 'qald' in filename:
            dataset = 'QALD 9'
        elif 'simple' in filename:
            dataset = 'SimpleQuestions'

        df[column] = df[column].astype('int64')

        avg = df.groupby(column).mean()
        count_df = df.groupby(column)['params.question'].count()
        avg = avg[['metrics.precision', 'metrics.recall', 'metrics.f1']]
        x = np.arange(avg.shape[0])
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.bar(x-width, avg['metrics.precision'], width, color='cyan')
        ax1.bar(x, avg['metrics.recall'], width, color='orange')
        ax1.bar(x + width, avg['metrics.f1'], width, color='green')
        ax1.set_xticks(x, avg.index)
        ax1.set_xlabel(r"Anzahl Triple", fontsize=18)
        ax1.legend(['Precision', 'Recall', 'F1-Score'], fontsize=18)
        # ax1.set_title(f'Performance von {filename.split(".")[0].split("_")[0]} auf {dataset} für verschiedener Named-Entity-Vorkommen')
        ax1.set_ylim(0,1)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(x, count_df)
        ax2.set_xticklabels(count_df)
        ax2.set_xlabel(r"Anzahl Fragen", fontsize=18)
        ax1.tick_params(labelsize=18)
        ax2.tick_params(labelsize=18)
        plt.savefig(f'../bar_plots_TRIPLE_count/{filename.split(".")[0].split("_")[0]}_{dataset}_bar_plot_TRIPLE_count.png')
        plt.close()
