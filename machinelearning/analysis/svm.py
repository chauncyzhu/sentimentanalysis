# coding=gbk
from sklearn import svm
import pandas as pd
import utils.newsgroup_path as npath
"""
    SVM的实现
"""
def svm_classification(train_data,train_target,test_data,test_target):
    clf = svm.SVC()
    clf.fit(train_data,train_target)

    result = clf.predict(test_data)
    count_eaq = 0  # 统计预测正确的结果个数
    count_not_eaq = 0
    for left, right in zip(result, test_target):
        print(left, right)
        if left == right:
            count_eaq += 1
        else:
            count_not_eaq += 1
    print(float(count_eaq) / len(test_target))
    print(count_not_eaq)

def newsgroup():
    # 数据读取
    pd_train = pd.read_csv(npath.TRAIN_BINARY_BDC_CSV)
    pd_test = pd.read_csv(npath.TEST_BINARY_BDC_CSV)

    #格式转换
    def f(x):
        return eval(x)
    train_data = list(pd_train['bdc'].apply(f))
    test_data = list(pd_test['bdc'].apply(f))

    def g(x):
        if x == '[1,0]':
            return 1
        else:
            return 0
    train_target = list(pd_train['class'].apply(g))
    test_target = list(pd_test['class'].apply(g))

    print(train_data)
    print(train_target)
    print(test_data)
    print(test_target)

    svm_classification(train_data, train_target, test_data, test_target)

if __name__ == '__main__':
    newsgroup()