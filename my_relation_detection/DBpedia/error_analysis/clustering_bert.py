from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from DBpedia.analysis.rl_evaluator_changed import load_gold_stardard, load_system_answers, \
    evaluate_dbpedia_returning_dataframe
from vis_clustering import kmeans_clustering, vis_elbow, vis_silhouette, vis_clustering, get_densest_cluster
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm

# BASE_MODEL = "distilbert-base-uncased"
DIRECTORY = 'all_items_short_list'
BASE_MODEL = f'..\\{DIRECTORY}\\results\\checkpoint-19955'

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
embedder = SentenceTransformer(BASE_MODEL)


def bert_vectorizer(text):
    # examples = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=256)
    # examples = tokenizer(text, padding='max_length')
    examples = embedder.encode(text)
    return examples


def exec_pca(X):
    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=100)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X)
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    return x0, x1


def get_clustering(X, num_clusters):
    kmeans, clusters = kmeans_clustering(X, num_clusters)

    cluster_labels = kmeans.labels_

    x0, x1 = exec_pca(X)

    return clusters, x0, x1, cluster_labels, kmeans.inertia_


def sub_clustering(df, cluster_number):
    sum_of_squared_distances = []
    silhouette_avg = []

    X = np.array(df['BERT_repr'].tolist())

    K = range(2,5)
    for num_clusters in K:
        df['cluster'], df['x0'], df['x1'], cluster_labels, inertia = get_clustering(X, num_clusters)

        silhouette_avg.append(silhouette_score(df[['x0', 'x1']], cluster_labels))
        sum_of_squared_distances.append(inertia)

        vis_clustering(df, num_clusters, title=f'BERT-Embeddings für Cluster #{cluster_number}')

        print(f'Cluster #{cluster_number}')
        clustering_df = df.groupby('cluster') \
            .agg({'f1': 'mean', 'question': 'size'}) \
            .rename(columns={'question': 'count'})

        print(clustering_df)

    vis_elbow(K, sum_of_squared_distances, title=f'BERT-Embeddings für Cluster #{cluster_number}')
    vis_silhouette(silhouette_avg, K, title=f'BERT-Embeddings für Cluster #{cluster_number}')


def main():
    pd.set_option('mode.chained_assignment', None)
    path_gold = f'..\\{DIRECTORY}\\eval\\gold.json'
    path_test = f'..\\{DIRECTORY}\\eval\\test.json'

    gold_answers = load_gold_stardard(path_gold)
    system_answers = load_system_answers(path_test)

    df = evaluate_dbpedia_returning_dataframe(gold_answers, system_answers)

    tqdm.pandas()
    df['BERT_repr'] = df['question'].progress_apply(lambda x: bert_vectorizer(x))

    sum_of_squared_distances = []
    silhouette_avg = []

    X = np.array(df['BERT_repr'].tolist())

    K = range(9, 10)

    for num_clusters in K:
        df['cluster'], df['x0'], df['x1'], cluster_labels, inertia = get_clustering(X, num_clusters)

        print('Clustering')
        clustering_df = df.groupby('cluster') \
            .agg({'f1': 'mean', 'question': 'size'}) \
            .rename(columns={'question': 'count'})

        print(clustering_df)

        # densest_cluster_df, cluster_number = get_densest_cluster(df, clustering_df)
        # sub_clustering(densest_cluster_df, cluster_number)

        silhouette_avg.append(silhouette_score(df[['x0', 'x1']], cluster_labels))
        sum_of_squared_distances.append(inertia)

        vis_clustering(df, num_clusters, alpha_cluster=1, alpha_f1=1, title=f'BERT-Embeddings', directory=DIRECTORY)
    vis_elbow(K, sum_of_squared_distances, title=f'BERT-Embeddings', directory=DIRECTORY)
    vis_silhouette(silhouette_avg, K, title=f'BERT-Embeddings', directory=DIRECTORY)


if __name__ == '__main__':
    main()