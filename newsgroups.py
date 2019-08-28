
__author__ = 'Giuseppe Serna'

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics

def newsgroups_classifier(maxdim=None, min_occurrences=1, remove_items=(), eng=None):

    newsgroups_train = fetch_20newsgroups(subset='train', remove=remove_items)
    vectorizer = CountVectorizer(min_df=min_occurrences, stop_words=eng, max_features=maxdim)
    vectorizer_boolean = CountVectorizer(binary=True, min_df=min_occurrences, stop_words=eng, max_features=maxdim)
    vectors = vectorizer.fit_transform(newsgroups_train.data)
    vectors_boolean = vectorizer_boolean.fit_transform(newsgroups_train.data)

    print vectors.shape

    newsgroups_test = fetch_20newsgroups(subset='test', remove=remove_items, random_state=rand_state)
    vectors_test = vectorizer.transform(newsgroups_test.data)
    vectors_test_boolean = vectorizer_boolean.transform(newsgroups_test.data)

    clf = MultinomialNB()
    clf2 = BernoulliNB()
    clf2.fit(vectors_boolean, newsgroups_train.target)
    clf.fit(vectors, newsgroups_train.target)
    pred = clf.predict(vectors_test)
    pred2 = clf2.predict(vectors_test_boolean)

    score = metrics.accuracy_score(newsgroups_test.target, pred)
    score_boolean = metrics.accuracy_score(newsgroups_test.target, pred2)
    # print vectorizer.stop_words_

    return score, score_boolean, vectors.shape[1]
