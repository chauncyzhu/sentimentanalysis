# coding=gbk
import math
import types
"""
    将词典转换为各种特征，其实神经网络常用的有tf、word2vec等，可以试一下bdc和df bdc
"""


def word2vec():

    pass


def getBDCVector(voca_dict,classnum,feature_name):
    """
    :param voca_dict: pandas dataframe
    :param classnum: 类别的个数，二分类个数为2，多分类则>2
    :param feature_name: 需要存放在哪一列
    :return: 不需要返回，因为传引
    """
    bdc_set = []
    # word_num[0]为单词，对每个词而言
    for index, row in voca_dict.iterrows():
        posibility_list = []  #词在每个类中的概率
        # 对每个分类而言
        for j in range(classnum):
            #__getOriginalValue将str转为list，这里是因为pandas从文件中读取dataframe时list会变成str
            if row['class_word_appear_set'][j] == 0:
                posibility_list.append(0)
            else:
                posibility_list.append(float(row['word_appear_set'][j]) / float(row['class_word_appear_set'][j]))
        temp = 0
        for j in range(classnum):
            try:
                temp += (posibility_list[j] / sum(posibility_list)) * (
                math.log(posibility_list[j] / sum(posibility_list)))
            except:
                print("error:",posibility_list[j],sum(posibility_list))
                temp += 0
        temp /= math.log(classnum)
        bdc_set.append(1 + temp)
    print(bdc_set)
    voca_dict[feature_name] = bdc_set

#计算新的想法中的df-bdc值，即p(d,ci)类i中出现词的文档数占总文档数量
#根据词典计算DFBDC值，voca_dict的格式为pandas dataframe，多分类时classnum>2，二分类时classnum=2（主要看voca_dict是什么样子的）
def getDFBDCVector(voca_dict,classnum,feature_name):
    df_bdc_set = []
    # word_num[0]为单词，对每个词而言
    for index, row in voca_dict.iterrows():
        posibility_list = []  # 词在每个类中的概率
        # 对每个分类而言
        for j in range(classnum):
            # __getOriginalValue将str转为list，这里是因为pandas从文件中读取dataframe时list会变成str
            if row['doc_class_set'][j] == 0:
                posibility_list.append(0)
            else:
                posibility_list.append(float(row['word_doc_set'][j]) / float(row['doc_class_set'][j]))
        temp = 0
        for j in range(classnum):
            try:
                temp += (posibility_list[j] / sum(posibility_list)) * (
                    math.log(posibility_list[j] / sum(posibility_list)))
            except:
                print("error:", posibility_list[j], sum(posibility_list))
                temp += 0
        temp /= math.log(classnum)
        df_bdc_set.append(1 + temp)
    print(df_bdc_set)
    voca_dict[feature_name] = df_bdc_set
