import sys

import numpy as np

sys.path.append('..')
from analysis.rl_evaluator_changed import evaluate_dbpedia_returning_dataframe, load_gold_stardard, load_system_answers
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os
import pandas as pd

file_path = ''
file_to_compare = ''


def get_series_single_result(df, f1_score_to_evaluate, feature):
    return df[feature][df['f1'] == f1_score_to_evaluate]


def get_series_two_results(df):
    return df['gold_relations_left']


def write_index_to_text(series, filename):
    with open(filename, 'w') as f:
        for index in series.index:
            f.write(index)
            f.write('\n')


def make_line_plot(series, gold):
    if gold:
        series = series.apply(lambda x: [item for sublist in x for item in sublist])
        series = series.explode().value_counts()
    else:
        series = series.explode().fillna('N/A').value_counts(dropna=False)
    plt.figure(figsize=(20, 8), dpi=150)
    plt.bar(series.index, series)
    plt.xticks(np.arange(0, len(series) + 1, 3), rotation=90)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.ylabel('Anzahl Vorkommen der Relation')
    plt.xlabel('Relationen')
    return series


def create_distribution_line_plot_gold_standard(df: pd.DataFrame, f1_score_to_evaluate):
    global file_path
    series = get_series_single_result(df, f1_score_to_evaluate, 'gold_relations')
    series = make_line_plot(series, True)
    plt.title(f'Verteilung der Gold Standards von Fragen mit F1-Score von {f1_score_to_evaluate}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{file_path}/error_analysis/distribution_line_plot_f1_{f1_score_to_evaluate}_gold_standard.png")
    write_index_to_text(series, f"{file_path}/error_analysis/distribution_line_plot_f1_{f1_score_to_evaluate}_gold_standard.txt")
    # for index in series.index:
    #     print(index)


def create_distribution_line_plot_system_output(df: pd.DataFrame, f1_score_to_evaluate):
    global file_path
    series = get_series_single_result(df, f1_score_to_evaluate, 'system_relations')
    series = make_line_plot(series, False)
    plt.title(f'Verteilung der Gold Standards von Fragen mit F1-Score von {f1_score_to_evaluate}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{file_path}/error_analysis/distribution_line_plot_f1_{f1_score_to_evaluate}_system_output.png")
    write_index_to_text(series, f"{file_path}/error_analysis/distribution_line_plot_f1_{f1_score_to_evaluate}_system_output.txt")


def create_distribution_line_plot_two_dfs(df: pd.DataFrame):
    global file_path
    series = get_series_two_results(df)
    series = make_line_plot(series, True)
    plt.title(f'Verteilung der Gold Standards von nicht korrekt vorhergesagten Fragen im Vergleich zweier Ergebnisse')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{file_to_compare}/error_analysis/distribution_line_plot_f1_compared_{file_path.split('.')[0].split('/')[-1]}.png")
    write_index_to_text(series, f"{file_to_compare}/error_analysis/distribution_line_plot_f1_compared_{file_path.split('.')[0].split('/')[-1]}.txt")


def get_df_from_filedialog():
    root = tk.Tk()
    root.withdraw()

    global file_path

    file_path = filedialog.askdirectory()
    path = os.path.join(file_path, 'error_analysis')

    print(path)

    os.makedirs(path, exist_ok=True)

    file_path_gold = f"{file_path}/eval/gold.json"
    file_path_test = f"{file_path}/eval/test.json"

    gold_standard = load_gold_stardard(file_path_gold)
    system_answers = load_system_answers(file_path_test)
    return evaluate_dbpedia_returning_dataframe(gold_standard, system_answers)


def save_df(df):
    global file_path
    df.to_csv(f"{file_path}/error_analysis/all_questions.csv")
    df[(df['f1'] != 1) & (df['precision'] != 1) & (df['recall'] != 1)].to_csv(f"{file_path}/error_analysis/all_questions_f1_ne_1.csv")
    df[(df['f1'] == 0)].to_csv(f"{file_path}/error_analysis/all_questions_f1_eq_0.csv")


def save_df_recall_ne_1(df):
    global file_path
    df[(df['recall'] != 1)].to_csv(f"{file_path}/error_analysis/all_questions_recall_ne_1.csv")

def analyse_single_result():
    df_1 = get_df_from_filedialog()
    save_df(df_1)
    create_distribution_line_plot_gold_standard(df_1, 1)
    create_distribution_line_plot_gold_standard(df_1, 0.5)
    create_distribution_line_plot_gold_standard(df_1, 0)

    create_distribution_line_plot_system_output(df_1, 0)


def analyse_single_result_recall_ne_1():
    df_1 = get_df_from_filedialog()
    save_df_recall_ne_1(df_1)

def analyse_two_results():
    global file_to_compare
    global file_path
    df_1 = get_df_from_filedialog()
    file_to_compare = file_path

    df_2 = get_df_from_filedialog()

    df_1_cleaned = df_1[(df_1['f1'] != 1) & (df_1['precision'] != 1) & (df_1['recall'] != 1)]
    df_2_cleaned = df_2[(df_2['f1'] != 1) & (df_2['precision'] != 1) & (df_2['recall'] != 1)]
    #
    # "Was funktioniert bei beiden gleichzeitig nicht?"
    int_df = pd.merge(df_1_cleaned, df_2_cleaned, how='inner', on=['question'], suffixes=('_left', '_right'))
    create_distribution_line_plot_two_dfs(int_df)
    int_df.to_csv(f"{file_to_compare}/error_analysis/no_correct_{file_path.split('.')[0].split('/')[-1]}.csv")
    #
    # # "Was funktioniert bei dem einen, aber bei dem anderen nicht?"
    # int_df = pd.merge(df_1, df_2, how='inner', on=['question'], suffixes=('_left', '_right'))
    # int_df = int_df[(int_df['precision_right'] != 1) & (int_df['recall_right'] != 1) & (int_df['f1_right'] != 1)]
    # int_df = int_df[(int_df['precision_left'] >= 0.5) & (int_df['recall_left'] >= 0.5) & (int_df['f1_left'] >= 0.5)]
    # int_df.to_csv(f"{file_to_compare}/error_analysis/left_correct_{file_path.split('.')[0].split('/')[-1]}.csv")
    # print(int_df)
    #
    # # "Was funktioniert bei beiden?"
    # int_df = pd.merge(df_1, df_2, how='inner', on=['question'], suffixes=('_left', '_right'))
    # int_df = int_df[(int_df['precision_right'] >= 0.5) & (int_df['recall_right'] >= 0.5) & (int_df['f1_right'] >= 0.5)]
    # int_df = int_df[(int_df['precision_left'] >= 0.5) & (int_df['recall_left'] >= 0.5) & (int_df['f1_left'] >= 0.5)]
    # int_df.to_csv(f"{file_to_compare}/error_analysis/both_f1_{file_path.split('.')[0].split('/')[-1]}.csv")
    # print(int_df)

    ####################################################################################################################
    #   DIESER ABSCHNITT IST FÜR VERGLEICHE MIT DREI ODER MEHR AUSGABEELEMENTEN GEDACHT

    # df_1_cleaned = df_1[(df_1['recall'] != 1)]
    # df_2_cleaned = df_2[(df_2['recall'] != 1)]
    #
    # int_df = pd.merge(df_1_cleaned, df_2_cleaned, how='inner', on=['question'])
    # create_distribution_line_plot_two_dfs(int_df)
    # int_df.to_csv(f"{file_to_compare}/error_analysis/recall_ne_1_{file_path.split('.')[0].split('/')[-1]}.csv")


def main():
    analyse_single_result()
    # analyse_two_results()
    # analyse_single_result_recall_ne_1()


if __name__ == '__main__':
    main()
