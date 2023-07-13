import pandas as pd
import os
from get_dataframe import get_dataframe
from get_NE_count import get_ne_count

if __name__ == '__main__':

    directory = '../csv/'
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filename = os.path.join(directory, filename)

        df = get_dataframe(filename)
        df.dropna(inplace=True)

        if ('lcquad' in filename) or ('qald' in filename):
            df['ne_count'] = df.apply(lambda row: get_ne_count(row['params.query_db']),
                                                 axis=1)
        elif ('simple' in filename):
            df['ne_count'] = 1

        avg = df.groupby('ne_count').mean()
        avg = avg[['metrics.precision', 'metrics.recall', 'metrics.f1']]
        # print(df)
        # f1 = df['metrics.f1'].mean()
        # prec = df['metrics.precision'].mean()
        # recall = df['metrics.recall'].mean()

        for index, row in avg.iterrows():

            print(f"""
            {filename}
            Named Entity Count: {index}
            precision: {row['metrics.precision']}
            recall: {row['metrics.recall']}
            f1-score: {row['metrics.f1']}
            """)