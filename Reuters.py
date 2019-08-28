from nltk.corpus import reuters
from random import shuffle

categories = [u'earn', u'acq', u'money-fx', u'grain', u'crude', u'trade', u'interest', u'ship', u'wheat', u'corn']
f_test = open('Test.txt', 'r')
test_names_array = f_test.read().split()
f_test.close()
test_target = {}
for i in categories:
    test_target[i] = []
test_data = []
for file in test_names_array:
    test_data.append(" ".join(reuters.words(file)))
    for cat in categories:
        if cat in reuters.categories(file):
            test_target[cat].append(1)
        else:
            test_target[cat].append(0)


def extract_reuters(percent):


    # Ora creo i data di training e i TrainTarget

    f_train = open('Train.txt', 'r')
    train_names_array = f_train.read().split()
    shuffle(train_names_array)
    selected_train_names_array = train_names_array[0:int(len(train_names_array)*percent)]
    f_train.close()
    train_target = {}
    for i in categories:
        train_target[i] = []
    train_data = []
    for file in selected_train_names_array:
        train_data.append(" ".join(reuters.words(file)))
        for cat in categories:
            if cat in reuters.categories(file):
                train_target[cat].append(1)
            else:
                train_target[cat].append(0)
    return train_data, train_target
