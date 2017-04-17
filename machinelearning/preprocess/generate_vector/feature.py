# coding=gbk
import math
import types
"""
    计算bdc值，向量或者数值
"""
def __getOriginalValue(value):
    if type(value) == types.StringType:
        return eval(value)
    else:
        return value

#根据词典计算BDC值，voca_dict的格式为pandas dataframe
def getBDCVector(voca_dict,classnum,feature_name):
    bdc_set = []
    # word_num[0]为单词，对每个词而言
    for index, row in voca_dict.iterrows():
        posibility_list = []  #词在每个类中的概率
        # 对每个分类而言
        for j in range(classnum):
            #__getOriginalValue将str转为list，这里是因为pandas从文件中读取dataframe时list会变成str
            posibility_list.append(float(__getOriginalValue(row['word_appear_set'])[j]) / float(__getOriginalValue(row['class_word_appear_set'])[j]))
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
            posibility_list.append(
                float(__getOriginalValue(row['word_doc_set'])[j]) / float(__getOriginalValue(row['doc_class_set'])[j]))
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

#将测试集中不存在的特征对应行为0
def getTotalVoca(pd_test,voca_dict):
    # 将pd_test中的特征加入voca_dict中，在这个操作中，应该将voca_dict中对应特征全部值设置为0
    for line in pd_test['content']:
        for word in line:
            if word not in voca_dict.index:
                voca_dict.loc[word] = 0  #添加对应的特征