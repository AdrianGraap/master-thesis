import pandas as pd
import os
from get_dataframe import get_dataframe
from get_question_type import check
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    directory = '../csv/'
    width = 0.2
    counter = 0

    files = ['EARL_question_wise_NEW_MODEL_lcquad.csv', 'FALCON_question_wise_lcquad.csv',
             'EARL_question_wise_NEW_MODEL_qald.csv', 'FALCON_question_wise_qald.csv']

    for file in files:
        counter += 1
        filename = os.fsdecode(file)
        if ('EARL' in filename):
            full_file = os.path.join(directory, filename)
            df_EARL = get_dataframe(full_file)
            df_EARL.dropna(inplace=True)
            df_EARL['question_type'] = df_EARL.apply(lambda row: check(row['params.question'], row['params.query_db']),
                                                     axis=1)
            avg_EARL = df_EARL.groupby('question_type').mean()
            count_df = df_EARL.groupby('question_type')['params.question'].count()
            avg_EARL = avg_EARL[['metrics.precision', 'metrics.recall', 'metrics.f1']]

        if ('FALCON' in filename):
            full_file = os.path.join(directory, filename)
            df_FALCON = get_dataframe(full_file)
            df_FALCON.dropna(inplace=True)
            df_FALCON['question_type'] = df_FALCON.apply(lambda row: check(row['params.question'], row['params.query_db']),
                                                     axis=1)
            avg_FACLON = df_FALCON.groupby('question_type').mean()
            count_df = df_FALCON.groupby('question_type')['params.question'].count()
            avg_FACLON = avg_FACLON[['metrics.precision', 'metrics.recall', 'metrics.f1']]

        if counter % 2 == 0:
            x = np.arange(avg_EARL.shape[0])
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()
            ax1.bar(x-0.5*width, avg_EARL['metrics.f1'], width, color='cyan')
            ax1.bar(x+0.5*width, avg_FACLON['metrics.f1'], width, color='green')
            ax1.set_xticks(x, avg_EARL.index)
            ax1.set_xlabel(r"Fragetypen", fontsize=18)
            ax1.set_ylabel('F1-Score', fontsize=18)
            ax1.legend(['EARL', 'FALCON'], fontsize=18)
            if 'lcquad' in filename:
                title_name = 'LC-QuAD 1.0'
            if 'qald' in filename:
                title_name = 'QALD 9'
            # ax1.set_title(f'Vergleich Performance der Systeme für verschiedene \nFragetypen auf {title_name}')
            ax1.set_ylim(0,1)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(x, count_df)
            ax2.set_xticklabels(count_df)
            ax2.set_xlabel(r"Anzahl Fragetypen", fontsize=18)
            # ax2.legend(fontsize=18)
            ax1.tick_params(labelsize=18)
            ax2.tick_params(labelsize=18)
            plt.savefig(f'../bar plots/{title_name}.png')
            plt.close()
