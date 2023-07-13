import numpy.linalg
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from get_dataframe import get_dataframe
import mpl_scatter_density
from get_question_type import check

y_axis = ['metrics.word_count', 'metrics.triple_count', 'metrics.triple_count_wikidata', 'metrics.triple_count_dbpedia']
x_axis = ['metrics.recall', 'metrics.f1', 'metrics.precision']
types = ['count', 'boolean', 'factoid', 'list']

if __name__ == '__main__':
    directory = '../csv/'
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        full_file = os.path.join(directory, filename)
        df = get_dataframe(full_file)

        if ('lcquad' in filename) or ('qald' in filename):

            df.dropna(inplace=True)
            df['question_type'] = df.apply(lambda row: check(row['params.question'], row['params.query_db']), axis=1)

            title = filename.split('.')[0]
            title = f'{title.split("_")[0]} {title.split("_")[3]}'

            for y in y_axis:
                fig = plt.figure(figsize=(10, 20))
                if 'word_count' in y:
                    param = 'Wortanzahl'
                elif 'triple_count' in y:
                    if 'wikidata' in y:
                        param = 'Anzahl Triple - Wikidata'
                    elif 'dbpedia' in y:
                        param = 'Anzahl Triple - DBPedia'
                    else:
                        param = 'Anzahl Triple'

                for i, question_type in enumerate(types):
                    if y in df:
                        plot_df = df[df['question_type'] == question_type]
                        ax = fig.add_subplot(4, 1, i + 1)
                        # plt.scatter(df[x], df[y])
                        try:
                            ax = sns.kdeplot(x=plot_df['metrics.f1'], y=plot_df[y], cmap="Reds", shade=True, bw_adjust=.5)
                        except numpy.linalg.LinAlgError:
                            pass
                        ax = sns.scatterplot(x=plot_df['metrics.f1'], y=plot_df[y], linewidth=0)
                        ax.set(xlabel='F1-Score', ylabel=param, title=f'Abhängigkeit Fragetyp "{question_type}" und F1-Score')

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
                    plt.suptitle(
                        f'Abhängigkeit {param} und verschiedene Metriken von {filename.split(".")[0].split("_")[0]}',
                        fontsize=20)
                    plt.savefig(f'../visual_clustering_QUESTION_TYPES/{filename.split(".")[0]}_{param}.png')
                    plt.close()
                    # plt.show()
