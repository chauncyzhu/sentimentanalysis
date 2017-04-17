# coding=gbk
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
import numpy as np

import preprocess.data_clean.reuters.import_data as rid
import preprocess.data_clean.newsgroup.import_data as nid  #数据获取
import utils.newsgroup_path as npath
import utils.reuters_path as rpath
"""
    贝叶斯的实现算法，主要还是用sklearn进行实现
"""
def naive_bayes(train_data,train_target,test_data,test_target):
    nbc = Pipeline([
        ('vect', TfidfVectorizer(

        )),
        ('clf', MultinomialNB(alpha=1.0)),
    ])
    nbc.fit(train_data, train_target)  # 训练我们的多项式模型贝叶斯分类器
    predict = nbc.predict(test_data)  # 在测试集上预测结果
    count_eaq = 0  # 统计预测正确的结果个数
    count_not_eaq = 0
    for left, right in zip(predict, test_target):
        print(left, right)
        if left == right:
            count_eaq += 1
        else:
            count_not_eaq += 1
    print(float(count_eaq) / len(test_target))
    print(count_not_eaq)


def newsgroup():
    CONFIRM_POS_CLASS = 0
    # 读取数据并转为dataframe
    pd_train, pd_test = nid.getTrainAndTest(npath.SOURCEFILE)  #注意这里已经变成了2分类

    # 应该根据指定的正类改变二分类中的class
    def f(x):
        if x[CONFIRM_POS_CLASS] == 1:
            return 1
        else:
            return 0

    pd_train['class'] = pd_train['class'].apply(f)
    pd_test['class'] = pd_test['class'].apply(f)
    print(pd_train)
    print(pd_test)

    train_data = list(pd_train['content'])
    train_target = list(pd_train['class'])
    test_data = list(pd_test['content'])
    test_target = list(pd_test['class'])

    naive_bayes(train_data, train_target, test_data, test_target)

def reuters():
    TOP_CLASS_NUM = 8
    CONFIRM_POS_CLASS = 0
    pd_train, pd_test = rid.getTrainAndTest(rpath.SOURCEFILE, TOP_CLASS_NUM)

    # 应该根据指定的正类改变二分类中的class
    def f(x):
        if x[CONFIRM_POS_CLASS] == 1:
            return 1
        else:
            return 0

    pd_train['class'] = pd_train['class'].apply(f)
    pd_test['class'] = pd_test['class'].apply(f)
    print(pd_train)
    print(pd_test)

    train_data = list(pd_train['content'])
    train_target = list(pd_train['class'])
    test_data = list(pd_test['content'])
    test_target = list(pd_test['class'])

    naive_bayes(train_data, train_target, test_data, test_target)

if __name__ == '__main__':
    #reuters()
    newsgroup()

