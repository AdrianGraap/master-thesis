import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from collections import OrderedDict
import os
import csv
from DBpedia.analysis.rl_evaluator_changed import load_gold_stardard, load_system_answers, \
    evaluate_dbpedia_returning_dataframe
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# initialize the vectorizer
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    # min_df=5,
    # max_df=0.95
)


def get_embeddings_for_tsv(row):
    return '\t'.join(str(emb) for emb in row)


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


def get_embeddings_for_tsv(row):
    return '\t'.join(str(emb) for emb in row)


def main():
    tqdm.pandas()

    path_gold = f'E:\\Hochschule\\Master\\master-thesis\\my_relation_detection\\DBpedia\\all_items_short_list\\eval\\gold.json'
    path_test = f'E:\\Hochschule\\Master\\master-thesis\\my_relation_detection\\DBpedia\\all_items_short_list\\eval\\test.json'

    gold_answers = load_gold_stardard(path_gold)
    system_answers = load_system_answers(path_test)

    df = evaluate_dbpedia_returning_dataframe(gold_answers, system_answers)

    df['cleaned'] = df['question'].progress_apply(lambda x: preprocess_text(x, remove_stopwords=True))

    vecs = vectorizer.fit_transform(df['cleaned'])

    vecs_array = vecs.toarray()
    df['vecs'] = ''
    df['vecs'] = df['vecs'].astype(object)
    for index, row in df.iterrows():
        df.at[index, 'vecs'] = vecs_array[index].tolist()

    X = df['vecs'].progress_apply(lambda x: get_embeddings_for_tsv(x))

    path_clustering = f'E:\\Hochschule\\Master\\master-thesis\\my_relation_detection\\DBpedia\\all_items_short_list\\tsv_embeddings'

    filepath = os.path.join(path_clustering)
    name = 'metadata_tfidf.tsv'
    with open(os.path.join(filepath, name), 'w+', encoding='utf-8') as file_metadata:
        file_metadata.write(f'question\tf1\n')
        for i, c in df.iterrows():
            row_content = f'{c["question"]}\t{c["f1"]}'
            file_metadata.write(row_content + '\n')

    name = 'embeddings_tfidf.tsv'
    with open(os.path.join(filepath, name), 'w+', encoding='utf-8') as tsvfile:
        for embedding in X:
            tsvfile.write(embedding + '\n')


if __name__ == '__main__':
    main()
