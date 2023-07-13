from DBpedia.analysis.rl_evaluator_changed import load_gold_stardard, load_system_answers, \
    evaluate_dbpedia_returning_dataframe
# import the dataset from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

# import other required libs
import pandas as pd
import numpy as np

# string manipulation libs
import re
import string
import nltk
from nltk.corpus import stopwords

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns

from vis_clustering import kmeans_clustering, vis_elbow, vis_silhouette, vis_clustering, exec_pca, get_densest_cluster, plot_dendrogram


# initialize the vectorizer
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    # min_df=5,
    # max_df=0.95
)


def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """

    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text


def get_clustering(X, num_clusters):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(X.toarray())

    x0, x1 = exec_pca(X)

    return model.labels_, x0, x1,  model


# def sub_clustering(df, cluster_number):
#     sum_of_squared_distances = []
#     silhouette_avg = []
#
#     X = vectorizer.fit_transform(df['cleaned'])
#     K = range(2,5)
#     for num_clusters in K:
#         df['cluster'], df['x0'], df['x1'], cluster_labels = get_clustering(X, num_clusters)
#
#         silhouette_avg.append(silhouette_score(df[['x0', 'x1']], cluster_labels))
#         sum_of_squared_distances.append(inertia)
#
#         vis_clustering(df, num_clusters, title=f'TF-IDF für Cluster #{cluster_number}')
#
#         print(f'Cluster #{cluster_number}')
#         clustering_df = df.groupby('cluster') \
#             .agg({'f1': 'mean', 'question': 'size'}) \
#             .rename(columns={'question': 'count'})
#
#         print(clustering_df)
#
#     vis_elbow(K, sum_of_squared_distances, title=f'TF-IDF für Cluster #{cluster_number}')
#     vis_silhouette(silhouette_avg, K, title=f'TF-IDF für Cluster #{cluster_number}')

def main():
    pd.set_option('mode.chained_assignment', None)
    path_gold = f'E:\\Hochschule\\Master\\master-thesis\\my_relation_detection\\DBpedia\\all_items_short_list\\eval\\gold.json'
    path_test = f'E:\\Hochschule\\Master\\master-thesis\\my_relation_detection\\DBpedia\\all_items_short_list\\eval\\test.json'

    gold_answers = load_gold_stardard(path_gold)
    system_answers = load_system_answers(path_test)

    df = evaluate_dbpedia_returning_dataframe(gold_answers, system_answers)
    df['cleaned'] = df['question'].apply(lambda x: preprocess_text(x, remove_stopwords=True))

    sum_of_squared_distances = []
    silhouette_avg = []

    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    X = vectorizer.fit_transform(df['cleaned'])

    K = range(2, 3)

    for num_clusters in K:
        # assign clusters and pca vectors to our dataframe
        df['cluster'], df['x0'], df['x1'], model = get_clustering(X, num_clusters)

        print('Clustering')
        clustering_df = df.groupby('cluster') \
              .agg({'f1': 'mean', 'question': 'size'}) \
              .rename(columns={'question': 'count'})

        print(clustering_df)

        # densest_cluster_df, cluster_number = get_densest_cluster(df, clustering_df)
        # sub_clustering(densest_cluster_df, cluster_number)

        # silhouette_avg.append(silhouette_score(df[['x0', 'x1']], df['cluster']))
        # sum_of_squared_distances.append(inertia)
        plot_dendrogram(model, truncate_mode="level", p=10)

        # vis_clustering(df, num_clusters, title="TF-IDF-Vektorisierung")
    # vis_elbow(K, sum_of_squared_distances, title='TF-IDF-Vektorisierung')
    # vis_silhouette(silhouette_avg, K, title='TF-IDF-Vektorisierung')


if __name__ == '__main__':
    main()
