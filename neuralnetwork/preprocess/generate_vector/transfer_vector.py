# coding=gbk
import numpy as np
import pandas as pd
"""
    将产生的字典转化为对应的文本向量，注意这里的文本都是序号，需要同时返回向量和对应的字典
"""
#转为序号向量，注意对于某些训练集和测试集，会出现NaN的情况，都用0来填充
def changeToFeatureVector(words,total_vova_value,name,target_file=None):
    total_vova_value[name] = list(range(1, len(total_vova_value) + 1))
    #将分词序列转换为向量
    def f(x,name):
        #print("name:",name)
        return list(pd.Series(total_vova_value[name][x]).fillna(0))  # 太神奇了，DataFrame允许列表输入，会自动迭代列表的每一个元素
    words[name] = words['content'].apply((lambda x:f(x,name)))  # 对于每一行words而言，但是神奇的是这个迭代居然到了每一个词？pn['words']相当于取出了一个series
    #将pn写入文件中
    if target_file:
        words.to_excel(target_file)