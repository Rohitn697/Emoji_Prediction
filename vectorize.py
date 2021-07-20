import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from naivebayes.NaiveBayes import NaiveBayes

df = pd.read_csv('data/cleaned_data.csv')
df = df.dropna()
corpus = df['text']

cv_uni = CountVectorizer(dtype=np.int8)
# cv_bi = CountVectorizer(dtype=np.int8, ngram_range=(2, 2))
#
# tfidf_uni = TfidfVectorizer()
# tfidf_bi = TfidfVectorizer(ngram_range=(2, 2))

X_Train, X_Test, Y_train, Y_test = train_test_split(corpus, df['emoji_label'], test_size=0.2, random_state=0)

X_train = cv_uni.fit_transform(X_Train)
X_test = cv_uni.transform(X_Test)

# X_train = cv_bi.fit_transform(X_Train)
# X_test = cv_bi.transform(X_Test)

# X_train = tfidf_uni.fit_transform(X_Train)
# X_test = tfidf_uni.transform(X_Test)

# X_train = tfidf_bi.fit_transform(X_Train)
# X_test = tfidf_bi.transform(X_Test)

'''
MNB = MultinomialNB()
MNB.fit(X_train, Y_train)
predict = MNB.predict(X_test)
confusion_matrix = metrics.confusion_matrix(predict, Y_test)
sns.heatmap(confusion_matrix)
plt.title('Confusion Matrix of Emoji Prediction\n using Multinomial Naive Bayes')
plt.xlabel('PREDICTED')
plt.ylabel('TEST DATA')
plt.show()
print(metrics.accuracy_score(predict, Y_test))
print(metrics.f1_score(predict, Y_test, average='macro'))
print(metrics.f1_score(predict, Y_test, average='micro'))
print(metrics.f1_score(predict, Y_test, average='weighted'))
'''

'''
classifier = MultinomialNB(alpha=0.1)
previous_score = 0
for alpha in np.arange(0,1,0.1):
  sub_classifier = MultinomialNB(alpha=alpha)
  sub_classifier.fit(X_train,Y_train)
  y_pred = sub_classifier.predict(X_test)
  score = metrics.accuracy_score(Y_test,y_pred)
  if score>previous_score:
        classifier=sub_classifier
  print("Alpha: {}, Score : {}".format(alpha,score))
'''
'''

