from __future__ import division
import newsgroups as ng
import newsgroupsrandom as rng
import ReutersClassifier as rc
import matplotlib.pyplot as plt


__author__ = 'Giuseppe Serna'

#Vengono impostati i parametri e chiamati i classificatori di 20newsgroup, infine salvate le learning curves

import numpy as np

num_examples = np.linspace(.1, 1.0, 10)
remove_items = ('headers', 'footers', 'quotes')
engwords = 'english'
y = {}
y_bool = {}
for i in num_examples:
    y[i] = []
    y_bool[i] = []
for i in range(0, 4):
    x = []
    print 'Call ' + str(i+1) + ' of 4 Classifier for 20NewsGroups'
    for n in num_examples:
        tmp = rng.newsgroups_classifier(n, 2, remove_items[0])
        y[n].append(tmp[0] * 100)
        x.append(tmp[2])
        y_bool[n].append(tmp[1] * 100)
print 'Merge the result...'
y_avg = []
y_bool_avg = []
for n in num_examples:
    y_avg.append(sum(y[n])/len(y[n]))
    y_bool_avg.append(sum(y_bool[n])/len(y_bool[n]))
name = '20NewsGroups'
plt.title(name)
plt.plot(x, y_avg, label='Multinomiale')
plt.plot(x, y_bool_avg, label='Bernoulli')
plt.grid(True)
plt.xlabel('Examples')
plt.ylabel('Classification accuracy')
plt.ylim(0, 100)
plt.xlim(0)
plt.legend(loc="best")
print 'I valori massimi di accuratezza sono ' + str(max(y_avg)) + '% per il Multinomiale e ' + str(max(y_bool_avg)) + '% per Bernoulli'


plt.savefig('Risultati 20newsgroups/' + name)
print 'The grapichs have been saved in Risultati 20Newsgroups directory'
plt.close()

#vengono impostati i parametri e chiamati i classificatori di reuter e le learning curves salvate
if True:

    num_examples = np.linspace(.1, 1.0, 10)
    categories = [u'earn', u'acq', u'money-fx', u'grain', u'crude', u'trade', u'interest', u'ship', u'wheat', u'corn']
    engwords = 'english'
    for cat in categories:
        y = []
        y_bool = []
        x = []
        print 'Start Reuters Classifier for category ' + cat
        for n in num_examples:
            tmp = rc.reuters_classifier(cat, n, 2)
            y.append(tmp[0] * 100)
            y_bool.append(tmp[1] * 100)
            x.append(tmp[2])
        print 'Finished  ' + cat
        name = cat
        plt.title(name)
        plt.plot(x, y, label='Multinomiale')
        plt.plot(x, y_bool, label='Bernoulli')
        plt.grid(True)
        plt.xlabel('Examples')
        plt.ylabel('Classification f1 score')
        plt.ylim(0, 100)
        plt.xlim(0)
        plt.legend(loc="best")
        plt.savefig('Risultati reuters/' + name)
        plt.close()
    print 'The grapichs of the 10 category have been saved in Risultati Reuters directory'
