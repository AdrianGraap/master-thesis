import pandas as pd
import os
from get_dataframe import get_dataframe

if __name__ == '__main__':

    directory = '../csv/'
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filename = os.path.join(directory, filename)

        df = get_dataframe(filename)
        # print(df)
        f1 = df['metrics.f1'].mean()
        prec = df['metrics.precision'].mean()
        recall = df['metrics.recall'].mean()

        print(f"""
        {filename}
        precision: {prec}
        recall: {recall}
        f1-score: {f1}
        """)