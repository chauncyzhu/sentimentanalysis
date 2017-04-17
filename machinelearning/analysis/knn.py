# coding=gbk
import numpy as np
import pandas as pd
import types
import time
from threading import Thread
from multiprocessing import Process,Manager
import multiprocessing
import math
"""
    找最近的K个邻居
"""
POOL_NUM = 2  #创建4个线程
# 用eval将str转为list
def getOriginalValue(value):
    if type(value) == types.StringType:
        return list(eval(value))
    else:
        return list(value)

#将dataframe数据等分成number份
def __getSplitPDData(pd_data,number):
    #对测试集分割成POOL_NUM份
    pd_data_len = len(pd_data)
    number = float(number)  #转成浮点数
    split_len = int(math.floor(pd_data_len/number))
    multi_pd_data = [pd_data[m:m+split_len] for m in range(pd_data_len) if m % split_len == 0]
    return multi_pd_data


#计算两个向量的距离，使用欧氏距离
def __calDistance(vector_one,vector_two):
    #转成numpy类型
    vector_one = np.array(list(vector_one))
    vector_two = np.array(list(vector_two))
    #维度置为一样
    max_len = np.max([len(vector_one),len(vector_two)])
    #在后面加上0
    vector_one = np.append(vector_one,[0]*(max_len-len(vector_one)))
    vector_two = np.append(vector_two, [0] * (max_len - len(vector_two)))
    return np.sqrt(np.sum(np.square(vector_one - vector_two)))

#KNN的核心，通过循环测试集来找到最近的K个训练及所属的分类，并用result包含起来，result是共享变量
def knn_core(pd_test,pd_train,k_list,feature_name,result):
    for index_test,row_test in pd_test.iterrows():
        #if index_test%200 == 0:
        print("has predict test doc num:",index_test)
        test_vector = eval(row_test[feature_name])
        test_class = eval(row_test['class']).index(1)  #因为每个文档只属于一个类，因此可以直接index
        #计算测试集和训练集的距离，并根据distance进行排序
        pd_distance = pd.DataFrame(data=[[[0],[0]]]*len(pd_train),columns=['class','distance'])   #第一个为所属类别，第二个为距离，index即对应的训练集序号
        #下面循环代码可以使用多线程
        for index_train,row_train in pd_train.iterrows():
            #if index_train % 1000 == 0:
                #print('has handle train doc num:', index_train, 'total num:', len(pd_train))
            pd_distance['class'][index_train] = eval(row_train['class'])
            #print("row_train['vector']",row_train['vector'])
            #print(eval(row_train['vector']))
            train_vector = eval(row_train[feature_name])
            pd_distance['distance'][index_train] = __calDistance(train_vector,test_vector)  #计算两个向量之间的距离
        #对distance_list进行处理
        pd_distance = pd_distance.sort_values(by='distance')  #指定列进行排序，需要返回值

        #取出最近的K个邻居
        neighbor = []
        for k in k_list:
            temp_np = np.array(pd_distance.head(k)['class']) #对应的类别
            neighbor.append(np.argmax(np.sum(temp_np,0)))  #sum沿着列进行求和，argmax找到最大值所在的位置
        #将结果封装
        result.append([test_class,neighbor])

#传入训练集和测试集，都是dataframe类型，并使用多线程来处理测试集
def knn(pd_train,pd_test,k_list,feature_name):
    #对于每个测试集
    begin = time.time()
    result = Manager().list()  #进程各自持有一份数据，默认无法共享数据，因此使用manager的list实现数据共享，注意，这个是在下面的append、get出错之后采取的第二种方法
    #result = list()  #由于Manager().list()，而result内部与顺序无关，因此直接append
    pool = multiprocessing.Pool()  # 创建4个进程，进程池设置最好等于CPU核心数量
    multi_data = __getSplitPDData(pd_test,POOL_NUM)  #等分成POOL_NUM份
    for data in multi_data:
        pool.apply_async(knn_core,(data, pd_train, k_list, feature_name, result))  # 注意，最好在真实的数据集上跑完整个，再在多进程中跑，因为多进程不会爆出完整的错误
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用，close()会等待池中的worker进程执行结束再关闭pool,而terminate()则是直接关闭
    pool.join()  # 等待进程池中的所有进程执行完毕
    end = time.time()
    print("total time:",end-begin)
    #对result进行处理，不知道为什么会是ListProxy object，用list()转为list
    return list(result)