import pandas as pd
import os


def get_dataframe(filename):
    if 'lcquad' in filename:
        cols = ['params.question', 'metrics.recall', 'metrics.f1', 'metrics.precision', 'metrics.triple_count',
                'metrics.word_count', 'params.question', 'params.query_db']
        df = pd.read_csv(filename, sep=',', usecols=cols)
        df = df[['metrics.precision', 'metrics.recall', 'metrics.f1', 'metrics.triple_count', 'metrics.word_count',
                 'params.question', 'params.query_db']]
    elif 'qald' in filename:
        if 'EARL' in filename:
            cols = ['params.question', 'metrics.recall', 'metrics.f1', 'metrics.precision',
                    'metrics.triple_count_dbpedia', 'metrics.word_count', 'params.question', 'params.query_db']
            df = pd.read_csv(filename, sep=',', usecols=cols)
            df = df[['metrics.precision', 'metrics.recall', 'metrics.f1', 'metrics.triple_count_dbpedia',
                     'metrics.word_count', 'params.question', 'params.query_db']]
        if 'FALCON' in filename:
            cols = ['params.question', 'metrics.recall', 'metrics.f1', 'metrics.precision',
                    'metrics.triple_count_wikidata',
                    'metrics.triple_count_dbpedia', 'metrics.word_count', 'params.question', 'params.query_db',
                    'params.query_wiki']
            df = pd.read_csv(filename, sep=',', usecols=cols)
            df = df[['metrics.precision', 'metrics.recall', 'metrics.f1', 'metrics.triple_count_dbpedia',
                     'metrics.triple_count_wikidata', 'metrics.word_count', 'params.question', 'params.query_db',
                     'params.query_wiki']]
    elif 'simple' in filename:
        cols = ['params.question', 'metrics.recall', 'metrics.f1', 'metrics.precision', 'metrics.word_count']
        df = pd.read_csv(filename, sep=',', usecols=cols)
        df = df[['metrics.precision', 'metrics.recall', 'metrics.f1', 'metrics.word_count', 'params.question']]
    elif 'SMART' in filename:
        cols = ['question', 'word_count', 'question_type', 'precision', 'recall', 'f1', 'relation_count']
        names = ['params.question', 'metrics.word_count', 'params.question_type', 'metrics.precision', 'metrics.recall',
                 'metrics.f1', 'metrics.relation_count']
        df = pd.read_csv(filename, sep=',', usecols=cols)
        df.rename(columns={'question': 'params.question', 'word_count': 'metrics.word_count',
                           'question_type': 'params.question_type', 'precision': 'metrics.precision',
                           'recall': 'metrics.recall', 'f1': 'metrics.f1',
                           'relation_count': 'metrics.relation_count'}, inplace=True)
        df = df.drop(df[df['params.question_type'] == 'error'].index)
        # df.set_axis(names, axis=1, inplace=True)
    return df
