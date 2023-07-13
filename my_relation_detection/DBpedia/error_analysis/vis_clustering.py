import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from matplotlib.markers import MarkerStyle
import umap
import umap.plot
from scipy.cluster.hierarchy import dendrogram

def kmeans_clustering(X, num_clusters=2):
    # initialize kmeans with n centroids
    kmeans = KMeans(n_clusters=num_clusters, random_state=100)
    # fit the model
    kmeans.fit(X)
    # store cluster labels in a variable
    clusters = kmeans.labels_
    return kmeans, clusters


def exec_pca(X):
    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=100)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X.toarray())

    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    return x0, x1


def vis_clustering(df, num_labels, title, directory, alpha_cluster=1, alpha_f1=1, max_f1=1):
    df = df[df['f1'] <= max_f1]

    # In diesem Abschnitt werden die Punkte aus den beiden auffälligen Cluster gefilter
    # FÜR TFIDF!!!
    # df = df[
    #         # (
    #         #     (df['x1'] >= 0.054) & (df['x1'] <= 0.133) & (df['x0'] >= -0.025) & (df['x0'] <= 0.00045) # oberes Cluster
    #         # )
    #     # | # ODER
    #         (
    #             (df['x1'] >= -0.034) & (df['x1'] <= 0.03) & (df['x0'] >= -0.07) & (df['x0'] <= -0.014)  # unteres Cluster
    #         )
    # ]
    #
    # print(df.describe())
    # df.to_csv(f'..\\all_items_short_list\\clustering\\unteres_cluster_alle.csv')
    ####################################################################################################################

    # set image size
    plt.figure(figsize=(12, 7), dpi=400)
    # set a title
    # plt.title(f'Textclustering mit {title} und maximalem F1-Score von {max_f1}', fontdict={"fontsize": 18})
    # set axes names
    plt.xlabel("X", fontdict={"fontsize": 20})
    plt.ylabel("Y", fontdict={"fontsize": 20})

    if title == 'TF-IDF-Vektorisierung':  # FÜR GLEICHE ACHSEN, WENN CLUSTER GEFILTERT WERDEN
        plt.xlim((-0.22235973785648774, 0.8398545093622579))
        plt.ylim((-0.3893828560832502, 0.6375412853292302))


    marker_cluster = MarkerStyle(marker='o', fillstyle='left')
    marker_f1 = MarkerStyle(marker='o', fillstyle='right')

    sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="tab10", alpha=alpha_cluster, marker=marker_cluster,
                    edgecolor='none', legend='full')
    f1 = sns.scatterplot(data=df, x='x0', y='x1', hue='f1', palette='Greys', alpha=alpha_f1, marker=marker_f1,
                         edgecolor='black')

    h, l = f1.get_legend_handles_labels()

    cluster_handles = h[0:num_labels]
    f1_handles = h[num_labels:]

    cluster_labels = l[0:num_labels]
    f1_labels = l[num_labels:]

    f1.get_legend().remove()
    # Create a legend for the first line.
    first_legend = f1.legend(handles=cluster_handles, labels=cluster_labels, loc='upper right', title='Cluster',
                             fontsize=18, markerscale=2, title_fontsize=18)
    #
    # # Add the legend manually to the Axes.
    f1.add_artist(first_legend)
    #
    # # Create another legend for the second line.
    f1.legend(handles=f1_handles, labels=f1_labels, loc='lower right', title='F1-Score',
              fontsize=18, markerscale=2, title_fontsize=18)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'..\\{directory}\\clustering\\{title}_{num_labels}__max_f1_{str(max_f1).replace(".", ",")}')


def vis_elbow(k, sum_of_squared_distances, title, directory):
    plt.clf()
    plt.plot(k, sum_of_squared_distances)
    plt.xlabel('Anzahl an Cluster')
    plt.ylabel('Summe des Quadrate der Abstände zu Clusterzentren')
    plt.title(f'Ellenbogen-Plot für Textclustering mit {title}')
    # set tick frequency
    plt.xticks(np.arange(min(k), max(k)+1, 1))
    # plt.show()
    plt.savefig(
        f'..\\{directory}\\clustering\\{title}_elbow_{min(k)}_{max(k)}'
    )

def vis_silhouette(silhouette_avg, k, title, directory):
    plt.clf()
    plt.bar(k, silhouette_avg)
    plt.xlabel('Anzahl der Cluster', fontsize=18)
    plt.ylabel('Silhouettenkoeffizient', fontsize=18)
    # plt.title(f'Silhouettenkoeffizient für Textclustering mit {title}')
    # set tick frequency
    plt.xticks(np.arange(min(k), max(k)+1, 1))
    # plt.show()
    plt.savefig(
        f'..\\{directory}\\clustering\\{title}_silhouette_{min(k)}_{max(k)}'
    )


def get_densest_cluster(df, clustering_df):
    max_count = clustering_df['count'].argmax()
    densest_cluster = df[df['cluster'] == max_count]
    return densest_cluster, max_count


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
