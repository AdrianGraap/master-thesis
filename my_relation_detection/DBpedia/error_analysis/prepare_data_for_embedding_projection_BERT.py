import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from collections import OrderedDict
import os
import csv
from DBpedia.analysis.rl_evaluator_changed import load_gold_stardard, load_system_answers, \
    evaluate_dbpedia_returning_dataframe
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# BASE_MODEL = 'distilbert-base-uncased'
BASE_MODEL = 'E:\\Hochschule\\Master\\master-thesis\\my_relation_detection\\DBpedia\\all_items_short_list\\results\\checkpoint-19955'
embedder = SentenceTransformer(BASE_MODEL)

def bert_vectorizer(text):
    # examples = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=256)
    # examples = tokenizer(text, padding='max_length')
    examples = embedder.encode(text)
    return examples


def get_embeddings_for_tsv(row):
    return '\t'.join(str(emb) for emb in row)


def main():
    tqdm.pandas()

    path_gold = f'E:\\Hochschule\\Master\\master-thesis\\my_relation_detection\\DBpedia\\all_items_short_list\\eval\\gold.json'
    path_test = f'E:\\Hochschule\\Master\\master-thesis\\my_relation_detection\\DBpedia\\all_items_short_list\\eval\\test.json'

    gold_answers = load_gold_stardard(path_gold)
    system_answers = load_system_answers(path_test)

    df = evaluate_dbpedia_returning_dataframe(gold_answers, system_answers)

    df['BERT_repr'] = df['question'].progress_apply(lambda x: bert_vectorizer(x))
    df.to_csv('cache.csv', sep=';')

    # df = pd.read_csv('cache.csv', sep=';', index_col=0, converters={'BERT_repr': lambda x: x.strip("[]").split(", ")})

    X = df['BERT_repr'].progress_apply(lambda x: get_embeddings_for_tsv(x))


    path_clustering = f'E:\\Hochschule\\Master\\master-thesis\\my_relation_detection\\DBpedia\\all_items_short_list\\tsv_embeddings'

    filepath = os.path.join(path_clustering)
    name = 'metadata_bert.tsv'
    with open(os.path.join(filepath, name), 'w+', encoding='utf-8') as file_metadata:
        file_metadata.write(f'question\tf1\n')
        for i, c in df.iterrows():
            row_content = f'{c["question"]}\t{c["f1"]}'
            file_metadata.write(row_content + '\n')

    name = 'embeddings_bert.tsv'
    with open(os.path.join(filepath, name), 'w+', encoding='utf-8') as tsvfile:
        for embedding in X:
            tsvfile.write(embedding + '\n')


if __name__ == '__main__':
    main()
