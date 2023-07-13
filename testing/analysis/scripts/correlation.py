import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from get_dataframe import get_dataframe

if __name__ == '__main__':
    directory = '../csv/'
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        full_file = os.path.join(directory, filename)
        df = get_dataframe(full_file)

        corr = df.corr()
        corr = corr.drop(['metrics.precision', 'metrics.recall', 'metrics.f1'])

        x_axis_labels = ['Precision', 'Recall', 'F1-Score']

        if 'lcquad' in filename:
            corr = corr.drop(['metrics.triple_count', 'metrics.word_count'], axis=1)
            y_axis_labels = ['Anzahl Triple', 'Wortanzahl']
            dataset = 'LC-QuAD 1.0'
        elif 'qald' in filename:
            if 'EARL' in filename:
                corr = corr.drop(['metrics.triple_count_dbpedia', 'metrics.word_count'],
                             axis=1)
                y_axis_labels = ['Anzahl Triple - DBPedia', 'Wortanzahl']
            if 'FALCON' in filename:
                corr = corr.drop(
                    ['metrics.triple_count_dbpedia', 'metrics.triple_count_wikidata', 'metrics.word_count'],
                    axis=1)
                y_axis_labels = ['Anzahl Triple - DBPedia', 'Anzahl Triple - Wikidata', 'Wortanzahl']
            dataset = 'QALD 9'
        elif 'simple' in filename:
            corr = corr.drop(['metrics.word_count'], axis=1)
            y_axis_labels = ['Wortanzahl']
            dataset = 'SimpleQuestions'
        elif 'SMART' in filename:
            corr = corr.drop(['metrics.word_count', 'metrics.relation_count'], axis=1)
            y_axis_labels = ['Anzahl Relationen', 'Wortanzahl']
            dataset = 'SMART'
        plt.figure(figsize=(12, 8))
        heat_map = sns.heatmap(corr, cmap="RdYlGn", annot=True, vmin=-1.0, vmax=1.0, xticklabels=x_axis_labels,
                               yticklabels=y_axis_labels)
        heat_map.set(title=f"Korrelationsmatrix - {filename.split('.')[0].split('_')[0]} auf Datensatz {dataset}")
        fig = heat_map.get_figure()
        fig.savefig(f'../heatmaps/{filename.split(".")[0]}.png')
