import tkinter as tk
from tkinter import filedialog
import pandas as pd
import json

with open('relation_count.json', encoding='utf-8') as f:
    data = json.load(f)

def get_csv_file():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()


    #, converters={'gold_relations_x': lambda x: x.replace("'", '"').json.load(x)}
    df = pd.read_csv(file_path, index_col=0, quotechar='"')
    if 'gold_relations_left' in df:
        column = 'gold_relations_left'
    else:
        column = 'gold_relations'
    df[column] = df[column].apply(lambda x: json.loads(x.replace("'", '"')))
    print(df)
    return df, file_path, column


def get_count(gold_relation):
    all_counts = []
    for sublist in gold_relation:
        count_list = []
        for relation in sublist:
            count_list.append(data[relation])
        all_counts.append(count_list)
    return all_counts


def main():
    df, file_path, column = get_csv_file()
    df['training_count'] = df.apply(lambda x: get_count(x[column]), axis=1)
    # print(df[['training_count', 'gold_relations_x']])
    df.to_csv(file_path)


if __name__ == '__main__':
    main()
