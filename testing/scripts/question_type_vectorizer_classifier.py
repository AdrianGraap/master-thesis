import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('../results/question_type.csv', index_col=0)

df['question'] = df['question'].apply(lambda x: ' '.join(x.split(' ')))

vectorizer = CountVectorizer(lowercase=False)

questions = df['question']
# y = df['question_type'].astype('int')
y = df['question_type']

questions_train, questions_test, y_train, y_test = train_test_split( questions, y, test_size=0.25, random_state=1000)

vectorizer.fit(questions_train)
X_train = vectorizer.transform(questions_train)
X_test = vectorizer.transform(questions_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

pickle.dump(vectorizer, open('../models/CountVectorizer.pickle', 'wb'))
pickle.dump(classifier, open('../models/Classifier.pickle', 'wb'))
