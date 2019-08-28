from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import random

__author__ = 'Giuseppe Serna'


def newsgroups_classifier(examples, min_occurrences=1, remove_items=(), eng=None):
    rand_state = random.randint(0, 100000)
    newsgroups = fetch_20newsgroups(subset='all', shuffle=True, remove=remove_items, random_state=rand_state)
    vectorizer = CountVectorizer(min_df=min_occurrences, stop_words=eng)
    vectorizer_boolean = CountVectorizer(binary=True, min_df=min_occurrences, stop_words=eng)
    vectors = vectorizer.fit_transform(newsgroups.data[0:(int(15077*examples))])
    vectors_boolean = vectorizer_boolean.fit_transform(newsgroups.data[0:(int(15077*examples))])

    vectors_test = vectorizer.transform(newsgroups.data[15077:18846])
    vectors_test_boolean = vectorizer_boolean.transform(newsgroups.data[15077:18846])

    clf = MultinomialNB()
    clf2 = BernoulliNB()
    clf2.fit(vectors_boolean, newsgroups.target[0:(int(15077*examples))])
    clf.fit(vectors, newsgroups.target[0:(int(15077*examples))])
    pred = clf.predict(vectors_test)
    pred2 = clf2.predict(vectors_test_boolean)

    score = metrics.accuracy_score(newsgroups.target[15077:18846], pred)
    score_boolean = metrics.accuracy_score(newsgroups.target[15077:18846], pred2)
    # print vectorizer.stop_words_
    return score, score_boolean, vectors.shape[0]
