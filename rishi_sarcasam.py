import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
data = pd.read_json('/home/richi/sarcasam_headlines/Sarcasm_Headlines_Dataset_v2.json', lines=True)
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
lsvc = LinearSVC()
lsvc.fit(features_train, labels_train)
lsvc.predict({'chandler': 'Chandler: Sounds like a date to me.', 
	'chandler 2': "Chandler: Alright, so I'm back in high school, I'm standing in the middle of the cafeteria, and I realize I am totally naked."})
print("train score for lsvc",lsvc.score(features_train, labels_train))
print("test score for lsvc",lsvc.score(features_test, labels_test))

# #model 2: using gaussian naive bayes
# gnb = GaussianNB()
# gnb.fit(features_train, labels_train)
# print("train score for gnb", gnb.score(features_train, labels_train))
# print("test score for gnb", gnb.score(features_test, labels_test))

# #model 3: using logistic regression
# logreg = LogisticRegression()
# logreg.fit(features_train, labels_train)
# print("train score for LogReg ", logreg.score(features_train, labels_train))
# print("test score for LogReg ", logreg.score(features_test, labels_test))

# #model 4: using random forest classifier
# rfc = RandomForestClassifier(n_estimators=10, random_state=0)
# rfc.fit(features_train, labels_train)
# print("train score for random forest", rfc.score(features_train, labels_train))
# print("test score for random forest", rfc.score(features_test, labels_test))
