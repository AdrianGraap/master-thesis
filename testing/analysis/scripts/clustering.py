import pandas as pd
from sklearn.cluster import KMeans
import os

if __name__ == '__main__':
    directory = '../csv/'
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filename = os.path.join(directory, filename)

        df = pd.read_csv(filename, sep=',', usecols=['question', 'f1', 'precision', 'recall'])

        feature_cols = ['f1', 'precision', 'recall']

        X = df.loc[:, feature_cols]

        kmeans = KMeans(n_clusters=3).fit(X)

        print(kmeans.labels_)
