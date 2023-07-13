import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from get_dataframe import get_dataframe
import mpl_scatter_density

y_axis = ['metrics.word_count', 'metrics.triple_count', 'metrics.triple_count_wikidata', 'metrics.triple_count_dbpedia',
          'metrics.relation_count']
x_axis = ['metrics.recall', 'metrics.f1', 'metrics.precision']

if __name__ == '__main__':
    directory = '../csv/'
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        full_file = os.path.join(directory, filename)
        df = get_dataframe(full_file)

        # title = filename.split('.')[0]
        # title = f'{title.split("_")[0]} {title.split("_")[3]}'

        for y in y_axis:
            fig = plt.figure(figsize=(10,15))
            if 'word_count' in y:
                param = 'Wortanzahl'
            elif 'triple_count' in y:
                if 'wikidata' in y:
                    param = 'Anzahl Triple - Wikidata'
                elif 'dbpedia' in y:
                    param = 'Anzahl Triple - DBPedia'
                else:
                    param = 'Anzahl Triple'
            elif 'relation_count' in y:
                param = 'Anzahl Relationen'
            for i, x in enumerate(x_axis):
                if 'recall' in x:
                    metric = 'Recall'
                elif 'precision' in x:
                    metric = 'Precision'
                elif 'f1' in x:
                    metric = 'F1-Score'

                if y in df:
                    ax = fig.add_subplot(3,1,i+1)
                    # plt.scatter(df[x], df[y])
                    ax = sns.kdeplot(x=df[x], y=df[y], cmap="Reds", shade=True, bw_adjust=.5)
                    ax = sns.scatterplot(x=df[x], y=df[y], linewidth=0)
                    ax.set(xlabel=metric, ylabel=param, title=f'Abhängigkeit {param} und {metric}')

                    # plt.xlabel(x.split('.')[1])
                    # plt.ylabel(y.split('.')[1])
                    # plt.title(title)
                    # # plt.show()
                    #
                    # sns.set_style('white')
                    # sns.kdeplot(x=df[x], y=df[y], cmap="Reds", shade=True, bw_adjust=.5)
            # plt.show()
                    # ax.figure.savefig(f'../visual_clustering/{filename.split(".")[0]}_{param}_{metric}.png')
            if y in df:
                if 'lcquad' in filename:
                    dataset = 'LC-QuAD 1.0'
                elif 'qald' in filename:
                    dataset = 'QALD 9'
                elif 'simple' in filename:
                    dataset = 'SimpleQuestions'
                plt.suptitle(f'Abhängigkeit {param} und verschiedene Metriken von {filename.split(".")[0].split("_")[0]}', fontsize=20)
                plt.savefig(f'../visual_clustering/{filename.split(".")[0]}_{param}.png')
                # plt.show()