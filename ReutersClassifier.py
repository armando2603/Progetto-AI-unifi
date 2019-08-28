import Reuters
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics


def reuters_classifier(cat, percent, min_occurrences=1, eng=None):

    train_data, train_target = Reuters.extract_reuters(percent)
    vectorizer = CountVectorizer(min_df=min_occurrences, stop_words=eng)
    vectorizer_boolean = CountVectorizer(binary=True, min_df=min_occurrences, stop_words=eng)
    vectors = vectorizer.fit_transform(train_data)
    vectors_boolean = vectorizer_boolean.fit_transform(train_data)

    vectors_test = vectorizer.transform(Reuters.test_data)
    vectors_test_boolean = vectorizer_boolean.transform(Reuters.test_data)

    clf = MultinomialNB()
    clf2 = BernoulliNB()
    clf2.fit(vectors_boolean, train_target[cat])
    clf.fit(vectors, train_target[cat])
    pred = clf.predict(vectors_test)
    pred2 = clf2.predict(vectors_test_boolean)

    x = metrics.f1_score(Reuters.test_target[cat], pred)
    y = metrics.f1_score(Reuters.test_target[cat], pred2)
    return x, y, vectors.shape[0]
