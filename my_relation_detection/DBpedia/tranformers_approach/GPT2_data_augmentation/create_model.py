from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import json
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
import numpy as np
import evaluate
import itertools
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
import torch
import pandas as pd

import sys
sys.path.append('..')
from count_and_map import count_and_map_all_items_short_list
from convert import to_jsonl

BASE_MODEL = "gpt2"
LEARNING_RATE = 1e-4
MAX_LENGTH = 256
BATCH_SIZE = 1
EPOCHS = 5

metric = evaluate.load("accuracy")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

def data_augmentation(train_data, all_data=True):
    train_data.set_format(type='pandas', columns=['question', 'relations', 'num_of_rels', 'question_type', 'mapped_relations', 'id'])
    train_data = train_data[:]
    if all_data:
        train_data = pd.concat([train_data, train_data, train_data])
    else:
        count_data = train_data[train_data['question_type'] == 'count']
        bool_data = train_data[train_data['question_type'] == 'boolean']
        two_rel_data = train_data[train_data['num_of_rels'] == 2]

        train_data = pd.concat([
            train_data,     # training data
            count_data,     # 10x count_data
            count_data,
            count_data,
            count_data,
            count_data,
            count_data,
            count_data,
            count_data,
            count_data,
            count_data,
            bool_data,      # 10x bool_data
            bool_data,
            bool_data,
            bool_data,
            bool_data,
            bool_data,
            bool_data,
            bool_data,
            bool_data,
            bool_data,
            two_rel_data,   # 5x 2-relation-data
            two_rel_data,
            two_rel_data,
            two_rel_data,
            two_rel_data
        ])
    # train_data[:].append(train_data[:]).append(train_data[:]).reset_format()
    # train_data.append(train_data).append(train_data)
    train_data = Dataset.from_pandas(train_data)
    return train_data

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    final_metrics = {}

    predictions = get_preds_from_logits(logits)

    # The global f1_metrics
    final_metrics["f1_micro"] = f1_score(labels, predictions, average="micro")
    final_metrics["f1_macro"] = f1_score(labels, predictions, average="macro")
    return final_metrics


def preprocess_function(examples):
    labels = [0] * len(id2label)
    for num in examples['mapped_relations']:
        labels[num] = 1
    # examples = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=256)
    examples = tokenizer(examples["question"])
    examples["label"] = labels
    # print(examples)
    return examples


def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    ret[:] = (logits[:] >= 0).astype(int)
    # ret = [1 if log >= 0 else 0 for log in logits[0]]
    print(ret)
    return ret


class MultiTaskClassificationTrainer(Trainer):
    def __init__(self, group_weights=None, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model_1, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model_1(**inputs)
        logits = outputs[0]
        labels = labels.float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

        return (loss, outputs) if return_outputs else loss


class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f"Epoch {state.epoch}: ")


# def preprocess_function(examples):
#     label = examples['mapped_relations']
#     # examples = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=256)
#     examples = tokenizer(examples["question"], padding=True)
#     examples["label"] = label
#     return examples


if __name__ == '__main__':
    with open('../SMART2022-RL-dbpedia-relation-vocabulary.json') as f:
        num_labels = len(json.load(f))
    print(num_labels)

    data, label2id = count_and_map_all_items_short_list()
    id2label = {value: key for key, value in label2id.items()}
    to_jsonl(data, filename='output.jsonl')

    data = load_dataset("json", data_files='output.jsonl', split='train')
    data = data.train_test_split(test_size=0.3)

    data['train'].to_json('data_train.json')
    data['test'].to_json('data_test.json')

    data['train'] = data_augmentation(data['train'], False)

    # print(data['train'][0])
    data = data.map(preprocess_function, batched=False, remove_columns=['question', 'relations', 'num_of_rels', 'id',
                                                                        'question_type', 'mapped_relations'])
    # tokenized_data = tokenized_data.rename_column('mapped_relations', 'label')
    # tokenized_data = tokenized_data.remove_columns(data["train"].column_names)

    # print(tokenized_data)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="f1_macro",
        load_best_model_at_end=True,
    )

    trainer = MultiTaskClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
