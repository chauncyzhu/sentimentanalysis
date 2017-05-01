# coding=gbk
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import cosine_similarity

def calDistance(pd_data,voca_dict,feature_name):
    """
    计算数据之间的cos相似度
    :param pd_data: 整个数据集
    :param feature_name: 对应的特征名
    :return: 返回相似度矩阵
    """
    ans = []
    for index,vector in pd_data.iterrows():
        temp = np.array(vector[feature_name]) * np.array(voca_dict[feature_name])
        ans.append(temp)

    sparseMatrix = sparse.csr_matrix(ans)   #转为稀疏矩阵
    similarities = cosine_similarity(sparseMatrix)
    print(similarities)
    return similarities


def knn_core(test_index,train_set,k_list):
    """
    knn分类的核心，对于每个测试集而言
    :param test_index: 测试数据对应的index，这里实际上是单个
    :param train_set: 训练集，包含有训练集和测试集的相似度矩阵，这里可能有改进的空间，因为train_set是对应所有的测试集
    :param k_list: 取出对应的k个值，这里需要改进，应该可以取出对应的k，这里直接是range(k)
    :return:
    """
    new_set = sorted(train_set, key=lambda file: file[2][test_index],reverse=True)
    #get top k list
    answer = []
    for k in k_list:
        topk_set = np.array([true_class[1] for true_class in new_set[0:k]])
        #直接sum topk_set
        answer.append(topk_set.sum(0).argmax())  #返回最大值所在的index，0即使正类，1即是负类
    return answer

def knn(train_index, test_index, distance, pd_data, k_list):
    """
    接收训练集角标，测试集角标，距离矩阵，模拟运行KNN算法
    :param train_index: 训练集角标，list
    :param test_index:  测试集角标，list
    :param distance: 距离矩阵
    :param pd_data: 数据集，这里需要取出对应的真实标签，'class'
    :param k_list: 取出对应的k个值
    :return: 宏平均微平均
    """
    ans_set = []
    #取出对应的真实标签和距离矩阵值
    train_set = [[index,pd_data['class'][index],distance[index]] for index in train_index]

    #对于每个测试集而言
    for i in range(len(test_index)):
        result = knn_core(test_index[i], train_set,k_list)  #对测试集进行分类，1即正类，0即负类
        ans_set.append([pd_data['class'][i].index(1),result])   #真实类所在的index，以及预测的类所在的index

    return ans_set

def evaluation_binaryclass(result_data,k_list):
    """
    二分类评估函数，默认class第一个是1为正类，第二个是1为负类
    :param result_data: knn返回的结果集，[test1[right,predict[k1,k2,k3]],test2[]]
    :param k_list: k list
    :return: [k1[p,r,f1],k2[p,r,f1],k3...]
    """
    print("classification result:",result_data)
    evaluation_result = []
    for k in range(len(k_list)):
        tp, fn, fp, tn = 0, 0, 0, 0
        for j in range(len(result_data)):
            # result_data[j][0]为真实标记，result_data[j][1][k]为第K个预测标记，均只有0，1两种，0为正（第0个数为1），1为负（第1个数为1）
            if result_data[j][0] == 0 and result_data[j][1][k] == 0:  # 真实==预测==正
                tp += 1
            if result_data[j][0] == 0 and result_data[j][1][k] == 1:  # 真实为正，预测为负
                fn += 1
            if result_data[j][0] == 1 and result_data[j][1][k] == 0:  # 真实为负，预测为正
                fp += 1
            if result_data[j][0] == 1 and result_data[j][1][k] == 1:  # 真实为负，预测为负
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
            f1 = 2*precision*recall/(precision+recall)
        evaluation_result.append([precision,recall,f1])
    return evaluation_result


def multi_flod(pd_data,voca_dict,k_list,flod_num,feature_name):
    """
    多折交叉实验，接收pandas dataframe，返回k_list p/r/f1
    :param pd_data: pandas dataframe，binary矩阵
    :param voca_dict: 字典列表，包含weighting
    :param k_list: knn中的k list
    :param flod_num: n折交叉实验法
    :param feature_name: 需要处理的特征，tfidf/tfrf/bdc
    :return: 
    """
    distance = calDistance(pd_data,voca_dict,feature_name)  #距离矩阵
    kf = KFold(len(pd_data), n_folds=flod_num)
    kf_evaluation_result = []   #[flod1[k1[p,r,f1],k2[p,r,f1],k3[p,r,f1]],flod2[k1,k2,k3]]
    for train_index, test_index in kf:  #对于每一折实验
        ans_set = knn(train_index, test_index,distance,pd_data, k_list)
        kf_evaluation_result.append(evaluation_binaryclass(ans_set,k_list))  #包含了若干个K，[k1[p,r,f1],k2[p,r,f1],k3[p,r,f1]]

    # each flod and each k
    kf_evaluation_result = np.array(kf_evaluation_result).T  #[p[k1[flod1,flod2],k2[flod1,flod2]],r[],f1[]]
    kf_evaluation_result = np.array([i.sum(1) for i in kf_evaluation_result]).T   #[k1[p_av,f_av],k2[p_av,f_av]]
    return pd.DataFrame(kf_evaluation_result)