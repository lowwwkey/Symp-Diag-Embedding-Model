# -*-coding:utf-8-*-

'''
@File       : classfication_algorithm.py
@Discription: 基于神经网络的症状/诊断嵌入模型
@Author     : Guangkai Li
@Date:      : 2017/10/10
'''

import heapq
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

def load_vec(select_num=100):
    global training_vec, training_label, test_vec, test_label
    training_vec = np.load('./data/training_vec.npy')
    training_label = np.load('./data/training_label.npy')
    test_vec = np.load('./data/test_vec.npy')
    test_label = np.load('./data/test_label.npy')


    mi = mutual_info_classif(training_vec, training_label, discrete_features='auto', n_neighbors=10, copy=True, random_state=3)
    mi = mi.tolist()
    max_mi_index = np.argsort(mi)[::-1][:select_num]
    max_mi_index = sorted(max_mi_index)
    training_vec = np.array([v[max_mi_index] for v in training_vec])
    test_vec = np.array([v[max_mi_index] for v in test_vec])

    X_train,y_train,X_test,y_test = [],[],[],[]
    for i in range(len(training_label)):
        if sum(training_vec[i]) > 0:
            X_train.append(training_vec[i])
            y_train.append(training_label[i])
    for i in range(len(test_label)):
        if sum(test_vec[i]) > 0:
            X_test.append(test_vec[i])
            y_test.append(test_label[i])

    training_vec, training_label, test_vec, test_label = X_train,y_train,X_test,y_test

def results(method):
	"""用5种评估指标评估模型
	
	Args:
		method: 5种分类模型
	"""
    model = method
    model.fit(training_vec, training_label)
    pre = model.predict(test_vec)
    accuracy = metrics.accuracy_score(test_label, pre)
    precision = metrics.precision_score(test_label, pre, average='weighted')
    recall = metrics.recall_score(test_label, pre, average='weighted')
    f1 = metrics.f1_score(test_label, pre, average='weighted')
    kappa = metrics.cohen_kappa_score(test_label, pre)
    return accuracy, precision, recall, f1, kappa


def cl():
    knn = KNeighborsClassifier(
        p=3, weights='distance', algorithm='auto', n_neighbors=5)
    lr = LogisticRegression(C=500.0, random_state=0)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=50)
    naive_bayes = GaussianNB()
    svm = SVC(gamma='auto', C=500)
    print("k近邻:%s,%s,%s,%s,%s" % results(knn))
    print("逻辑斯蒂回归:%s,%s,%s,%s,%s" % results(lr))
    print("决策树:%s,%s,%s,%s,%s" % results(tree))
    print("朴素贝叶斯:%s,%s,%s,%s,%s" % results(naive_bayes))
    print("支持向量机:%s,%s,%s,%s,%s" % results(svm))


if __name__ == '__main__':
    load_vec()
    cl()
