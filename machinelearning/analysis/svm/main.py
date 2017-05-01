# coding=gbk
"""
    SVM的main函数调用
"""
import time
import numpy as np
import pandas as pd
import machinelearning.analysis.svm.svm as svm
import utils.sentiment_data_path as sdp

#获得二分类数据，只需要改动class列就行
def get_binary_class_data(train_csv,test_csv,voca_csv,feature_name):
    # 数据读取，首先读取序号向量，试一下comment数据
    print("begin reading")
    mylist = []
    for chunk in pd.read_csv(train_csv,index_col=0,chunksize=2000,encoding="utf8"):
        mylist.append(chunk)
    pd_train = pd.concat(mylist,axis=0,ignore_index=True)  #如果为0的话，就是0，角标从1开始
    del mylist

    pd_test = pd.read_csv(test_csv,index_col=0)

    #控制数据量
    pd_train = pd.concat([pd_train.head(100),pd_train.tail(100)],ignore_index=True)
    pd_test = pd.concat([pd_test.head(20),pd_test.tail(20)],ignore_index=True)

    # 读取字典
    voca_dict = pd.read_csv(voca_csv,index_col=0)

    print("end reading.")

    print("begin transfer to "+feature_name+" vector....")
    def g(x):
        if eval(x) == [0,1]:
            return 1  #负类
        elif eval(x) == [1,0]:
            return 0  #正类
        else:
            return -1  #错误

    pd_train['class'] = pd_train['class'].apply(g)
    pd_test['class'] = pd_test['class'].apply(g)

    def f(x):
        return np.array(eval(x) * np.array(voca_dict[feature_name]))
    pd_train['sequence'] = pd_train['sequence'].apply(f)
    pd_test['sequence'] = pd_test['sequence'].apply(f)

    print("end transfer to "+feature_name+" vector.")

    # 数据量控制
    #pd_test = pd_test.head(10)
    return (pd_train,pd_test,voca_dict)

def main(train_csv,test_csv,voca_csv,feature_name,evaluation_csv=None):
    # 对于每个测试集
    begin = time.time()
    pd_train, pd_test, voca_dict = get_binary_class_data(train_csv, test_csv, voca_csv, feature_name)
    #结果集
    result = svm.svm_classification(list(pd_train['sequence']),list(pd_train['class']),list(pd_test['sequence']),list(pd_test['class']))
    evaluation_result = svm.evaluation_binaryclass(result)
    end = time.time()
    print("result:",evaluation_result)
    if evaluation_csv:
        evaluation_result.to_csv(evaluation_csv,encoding="utf8")
    print("total time:", end - begin)

if __name__ == '__main__':
    # 导入数据
    feature_name = "tf_idf"
    train_csv = sdp.TRAIN_BINARY_COMMENT
    test_csv = sdp.TEST_BINARY_COMMENT
    voca_csv = sdp.VOCA_BINARY_COMMENT
    evaluation_csv = sdp.EVALUATION_SVM_COMMENT
    main(train_csv, test_csv, voca_csv,feature_name, evaluation_csv=evaluation_csv)

