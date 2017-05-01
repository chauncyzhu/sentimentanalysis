# coding=gbk
from sklearn import svm
import pandas as pd
import numpy as np
"""
    SVM的实现
"""
def svm_classification(train_data,train_target,test_data,test_target):
    clf = svm.SVC()
    clf.fit(train_data,train_target)

    result = clf.predict(test_data)

    return np.array([test_target,result]).T

def evaluation_binaryclass(result_data):
    """
    二分类评估函数，默认class第一个是1为正类，第二个是1为负类
    :param result_data: knn返回的结果集，[test1[right,predict[k1,k2,k3]],test2[]]
    :param k_list: k list
    :return: [k1[p,r,f1],k2[p,r,f1],k3...]
    """
    print("classification result:",result_data)
    tp, fn, fp, tn = 0, 0, 0, 0
    for j in range(len(result_data)):
        # result_data[j][0]为真实标记，result_data[j][1][k]为第K个预测标记，均只有0，1两种，0为正（第0个数为1），1为负（第1个数为1）
        if result_data[j][0] == 0 and result_data[j][1] == 0:  # 真实==预测==正
            tp += 1
        if result_data[j][0] == 0 and result_data[j][1] == 1:  # 真实为正，预测为负
            fn += 1
        if result_data[j][0] == 1 and result_data[j][1] == 0:  # 真实为负，预测为正
            fp += 1
        if result_data[j][0] == 1 and result_data[j][1] == 1:  # 真实为负，预测为负
            tn += 1
    if tp + fp == 0:
        precision = float(0)
    else:
        precision = float(tp) / (tp + fp)
    if tp + fn == 0:
        recall = float(0)
    else:
        recall = float(tp) / (tp + fn)
    if precision + recall == 0:
        f1 = float(0)
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return pd.DataFrame([precision, recall, f1])
