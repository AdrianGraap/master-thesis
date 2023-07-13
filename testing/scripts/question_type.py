import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

vectorizer = pickle.load(open('../models/CountVectorizer.pickle', 'rb'))
classifier = pickle.load(open('../models/Classifier.pickle', 'rb'))

def check_count(question):
    count_list = ['count']
    if 'how many' in question.lower():
        return True
    if question.lower().split(' ')[0] in count_list:
        return True

    return False


def check_boolean(question):
    bool_list = ['was', 'is', 'are', 'does', 'did']
    q_list = question.lower().split(' ')
    return q_list[0] in bool_list


def check_factoid(question):
    word_list = ['what', 'when', 'which', 'who', 'how', 'where', 'whom', 'whose', 'why', 'whats']
    return (question.split(' ')[0].lower() in word_list) or (question.split(' ')[1].lower() in word_list)


def check_list(question):
    list_list = ['give me', 'name', 'list', 'tell me', 'give']
    q_list = question.lower().split(' ')
    q = ' '.join(q_list)
    if q_list[0] in list_list:
        return True
    if ' '.join(q_list[0:2]) in list_list:
        return True
    return False


def check_rest(question):
    return classifier.predict(vectorizer.transform([question]))[0]


def check_quest(quest):
    try:
        if check_count(quest):
            return 'count'
        elif check_boolean(quest):
            return 'boolean'
        elif check_factoid(quest):
            return 'factoid'
        elif check_list(quest):
            return 'list'
        else:
            return check_rest(quest)
    except IndexError:
        return 'error'
