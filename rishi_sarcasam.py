import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
data = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)
print(data.head())
print(data.isnull().any(axis=0))

#cleaning the data
data['headline'] = data['headline'].apply(lambda s : re.sub('[^a-zA-Z]',' ', s))

#feature and labeling
features = data['headline']
labels = data['is_sarcastic']

#stemming
ps = PorterStemmer()

features = features.apply(lambda x: x.split())
features = features.apply(lambda x: ' '.join([ps.stem(word) for word in x]))

#vectorizing the data using tf-idf(term frequency-inverse document frequency)
tv = TfidfVectorizer(max_features=5000)
features = list(features)
features = tv.fit_transform(features).toarray()

#train test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.05, random_state=0)

#model 1: using linear SVM
lsvc = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', dual=False), cv=5)
lsvc.fit(features_train, labels_train)
print("train score for lsvc",lsvc.score(features_train, labels_train))
print("test score for lsvc",lsvc.score(features_test, labels_test))

#model 2: using gaussian naive bayes
gnb = GaussianNB()
gnb.fit(features_train, labels_train)
print("train score for gnb", gnb.score(features_train, labels_train))
print("test score for gnb", gnb.score(features_test, labels_test))

#model 3: using logistic regression
logreg = LogisticRegression()
logreg.fit(features_train, labels_train)
print("train score for LogReg ", logreg.score(features_train, labels_train))
print("test score for LogReg ", logreg.score(features_test, labels_test))

#model 4: using random forest classifier
rfc = RandomForestClassifier(n_estimators=10, random_state=0)
rfc.fit(features_train, labels_train)
print("train score for random forest", rfc.score(features_train, labels_train))
print("test score for random forest", rfc.score(features_test, labels_test))


#predict 1: using Linear SVM model
data = pd.read_csv('friends_dataset.csv', dtype={'Text': 'str'})
data.dropna(inplace=True)
print(data.info())
print(data.head())
print(data.isnull().any(axis=0))
features = data['Text'].apply(lambda s: re.sub('[^a-zA-Z]', ' ', s))
features = features.apply(lambda x: x.split())
features = features.apply(lambda x: ' '.join([ps.stem(word) for word in x]))
features = list(features)
features = tv.fit_transform(features).toarray()
data['sarcastic'] = lsvc.predict(features)
data = data.replace({0: 'non-sarcastic', 1: 'sarcastic'})
data = data.groupby('Speaker')['sarcastic'].value_counts().unstack().fillna(0)
data['sarcasm ratio'] = data['non-sarcastic'] / data['sarcastic']
print(data)

#predict 2: take command-line input and predict using model
print("##########################################")
print()
print("How sarcastic are you?!")
print()
print("##########################################")
print()
while(True):
    print("Tell me something ...")
    inputString = str(input())
    data = pd.DataFrame({'Text': [inputString]}, dtype=str)
    data.dropna(inplace=True)
    # print(data.info())
    # print(data.head())
    # print(data.isnull().any(axis=0))
    features = data['Text'].apply(lambda s: re.sub('[^a-zA-Z]', ' ', s))
    features = features.apply(lambda x: x.split())
    features = features.apply(lambda x: ' '.join([ps.stem(word) for word in x]))
    features = list(features)
    features = tv.transform(features).toarray()
    sarcasm = lsvc.predict_proba(features)[:,1][0]
    sarcasm = sarcasm * 100
    sarcasm = "{:.2f}".format(sarcasm)
    print("That was " + sarcasm + "% sarcastic!")
    print()
