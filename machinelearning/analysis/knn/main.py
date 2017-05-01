# coding=gbk
"""
    knn函数的调用
"""
import time
import numpy as np
import pandas as pd
import machinelearning.analysis.knn.knn_sklearn as knn
import utils.sentiment_data_path as sdp


#默认参数
K_LIST = 50
FLOD_NUM = 5  #五折交叉实验
CONFIRM_POS_CLASS = 0  #指定二分类正类序号

#获得二分类数据，只需要改动class列就行
def get_binary_class_data(train_csv,test_csv,voca_csv,feature_name):
    # 数据读取，首先读取序号向量，试一下comment数据
    print("begin reading")
    pd_train = pd.read_csv(train_csv,index_col=0)  #如果为0的话，就是0，角标从1开始
    pd_test = pd.read_csv(test_csv,index_col=0)

    #控制数据量
    pd_train = pd_train.head(100)
    pd_test = pd_test.head(20)

    # 读取字典
    voca_dict = pd.read_csv(voca_csv,index_col=0)

    print("end reading.")

    # def f(x):
    #     x = (pd.Series(eval(x))-1).replace(-1,np.nan)  #将其转为series序列，需要减去1
    #     voca_index =  list(pd.Series(voca_dict.index)[x])  #对于nan则不进行操作，这里需要转为list，否则会出现reindex的错误
    #     vector = np.array(voca_dict[feature_name][voca_index].fillna(0))
    #     #print("vector.nonzero():",vector.nonzero())
    #     return vector

        # x_voca = []
        # for i in eval(x):
        #     voca = voca_dict.index[int(i)-1]
        #     if int(i) == 0:  #注意有一个字典居然是NULL，这里单独列出来，替换为了“保留”
        #         x_voca.append(0)
        #     else:
        #         x_voca.append(voca_dict[feature_name][voca])  #会出现nan，pandas无法识别
        # return x_voca


    print("begin transfer to "+feature_name+" vector....")
    pd_train['class'] = pd_train['class'].apply(eval)
    pd_test['class'] = pd_test['class'].apply(eval)
    pd_train[feature_name] = pd_train['sequence'].apply(eval)
    pd_test[feature_name] = pd_test['sequence'].apply(eval)
    print("end transfer to "+feature_name+" vector.")

    # 数据量控制
    #pd_test = pd_test.head(10)
    return (pd_train,pd_test,voca_dict)

def main(pd_data,voca_dict, k_num,flod_num,feature_name,evaluation_csv=None):
    # 对于每个测试集
    begin = time.time()
    kf_evaluation_result = knn.multi_flod(pd_data,voca_dict, k_num,flod_num,feature_name)
    end = time.time()
    print("result:",kf_evaluation_result)
    if evaluation_csv:
        kf_evaluation_result.to_csv(evaluation_csv,encoding="utf8")
    print("total time:", end - begin)

if __name__ == '__main__':
    #导入数据
    feature_name = "tf_idf"
    train_csv = sdp.TRAIN_BINARY_COMMENT
    test_csv = sdp.TEST_BINARY_COMMENT
    voca_csv = sdp.VOCA_BINARY_COMMENT
    evaluation_csv = sdp.EVALUATION_KNN_COMMENT

    pd_train, pd_test, voca_dict = get_binary_class_data(train_csv,test_csv,voca_csv,feature_name)
    pd_data = pd.concat([pd_train,pd_test],ignore_index=True)
    main(pd_data,voca_dict, K_LIST, FLOD_NUM, feature_name,evaluation_csv=evaluation_csv)