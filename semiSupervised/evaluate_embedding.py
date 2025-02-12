import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import warnings


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def logistic_classify(x, y):
    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    accs_val = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):

        # test
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls = torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()

        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls = torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()

        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs_val.append(acc.item())

    return np.mean(accs_val), np.mean(accs)


def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)


def randomforest_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    ret = np.mean(accuracies)
    return np.mean(accuracies_val), ret


def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)


def semi_supervised_svc_classify(x, y, search, labeled_ratio=0.1):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []

    for train_index, test_index in kf.split(x, y):
        # 划分训练集和测试集
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 仅选取部分数据作为有标签数据
        num_labeled = int(len(y_train) * labeled_ratio)
        labeled_indices = np.random.choice(range(len(y_train)), num_labeled, replace=False)

        x_labeled = x_train[labeled_indices]
        y_labeled = y_train[labeled_indices]

        # 训练初始模型
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10, probability=True)
        classifier.fit(x_labeled, y_labeled)

        # 生成伪标签
        pseudo_labels = classifier.predict(x_train)

        # 使用伪标签训练最终模型
        x_combined = np.vstack((x_labeled, x_train))
        y_combined = np.concatenate((y_labeled, pseudo_labels))

        classifier.fit(x_combined, y_combined)

        # 测试集评估
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # 验证集划分
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if i not in test_index]

        x_train, x_val = x[train_index], x[test_index]
        y_train, y_val = y[train_index], y[test_index]

        # 重新训练模型
        if search:
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10, probability=True)
        classifier.fit(x_train, y_train)

        accuracies_val.append(accuracy_score(y_val, classifier.predict(x_val)))

    return np.mean(accuracies_val), np.mean(accuracies)


def evaluate_embedding(embeddings, labels, search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc = 0
    acc_val = 0

    '''
    _acc_val, _acc = logistic_classify(x, y)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    '''

    _acc_val, _acc = svc_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc

    """
    _acc_val, _acc = linearsvc_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    """
    '''
    _acc_val, _acc = randomforest_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    '''

    print(acc_val, acc)

    return acc_val, acc


def semi_evaluate_embedding(embeddings, labels, search=True, label_ratio=0.1):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc = 0
    acc_val = 0

    _acc_val, _acc = semi_supervised_svc_classify(x, y, search, label_ratio)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc

    print(acc_val, acc)

    return acc_val, acc

'''
if __name__ == '__main__':
    evaluate_embedding('./data', 'ENZYMES', np.load('tmp/emb.npy'))
'''
