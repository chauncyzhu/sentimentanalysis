# coding=gbk
import numpy as np
import pandas as pd
"""
    将产生的字典转化为对应的文本向量，注意这里的文本都是序号，需要同时返回向量和对应的字典
"""
def changeToFeatureVector(words,total_vova_value,target_file=None):
    """
    转为序号向量，注意对于某些训练集和测试集，会出现NaN的情况，都用0来填充
    :param words: 需要转换的数据
    :param total_vova_value: 总的词语
    :param target_file:需要写入的文件目录，是把
    :return:
    """
    print("begin change to feature vector.")
    total_vova_value["sequence"] = list(range(1, len(total_vova_value) + 1))  #序号向量
    #将分词序列转换为向量
    def f(x):
        return list(pd.Series(total_vova_value["sequence"][x]).fillna(0))  # 太神奇了，DataFrame允许列表输入，会自动迭代列表的每一个元素
    words["sequence"] = words['content'].apply(f)  # 对于每一行words而言，但是神奇的是这个迭代居然到了每一个词？pn['words']相当于取出了一个series
    #将pn写入文件中
    if target_file:
        words.to_csv(target_file,encoding="utf8")
    print("end change to feature vector.")

#将pd_data的content属性转为特征向量，放在feature_name这一列中，targetfile可以指定要保存在那个文件里面去
#这里的vector是和特征向量的大小一样
def changeToBinaryVector(pd_data,voca_dict,target_file=None):
    print("voca_dict:",voca_dict)
    # 将分词序列转换为向量
    def f(x):
        vector = []  # vector长度实际
        for word in voca_dict.index:
            if word in x:  #如果特征在文档中
                vector.append(1)
            else:
                vector.append(0)  #会存在NaN
        return vector
    pd_data["sequence"] = pd_data['content'].apply(f)  # 对于每一行words而言，但是神奇的是这个迭代居然到了每一个词？pn['words']相当于取出了一个series
    #将NaN转成0
    #pd_data['vector'] = pd_data['vector'].apply(lambda x:list(np.nan_to_num(x)))
    # 将pn写入文件中
    if target_file:
        pd_data.to_csv(target_file,encoding="utf8")
