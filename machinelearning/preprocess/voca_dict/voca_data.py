# coding=gbk
import pandas as pd
import time
import copy
import numpy as np
"""
    找出数据预处理的dataframe对应的字典列表
"""
#获取data对应的字典，可以在这里对低频词做一个筛选
def getUniqueVocabulary(data):
    w = []  # 将所有词语整合在一起
    for i in data['content']:
        w.extend(i)
    vocabulary = pd.DataFrame(pd.Series(w).value_counts(),columns=['tf'])  #这里就隐含了tf值
    vocabulary = vocabulary[vocabulary['tf'] > 10]   #去掉某些低频词
    vocabulary = vocabulary[vocabulary['tf'] < 4000]  # 去掉某些高频词
    return vocabulary  #pd.DataFrame

#传入需要统计相关值的dataframe和相对应的字典数据
def getRelativeValue(pd_data,voca_data,class_num):
    #get:
    # number of all vocabulary in a class [类1总词数，类2总词数,.....]
    # number of every vocabulary in every class [[类1该词语出现数,类2该词语出现数...]，[类1该词语出现数,类2该词语出现数...],...]
    # number of documents contains every vocabulary in every class [[类1出现该词语的文档数,类2出现该词语的文档数...]，[类1出现该词语的文档数,类2出现该词语的文档数...],...]
    # number of documents in every class [类1总文档数，类2总文档数,.....]
    #but first, we need all vocabulary list
    #所有的词汇
    voca = voca_data.index
    temp = [0]*class_num
    word_appear_set = []
    class_word_appear_set = copy.deepcopy(temp)
    word_doc_set = []
    doc_class_set = copy.deepcopy(temp)
    for i in voca:
        word_appear_set.append(copy.deepcopy(temp))
        word_doc_set.append(copy.deepcopy(temp))
    print 'calculating'
    begin = time.time()
    #对每一份文档进行循环
    for index, row in pd_data.iterrows():
        if(index%100==0):
            print("now doc num:",index)
        content = row['content']
        class_list = row['class']
        for i in range(len(voca)):
            if voca[i] in content:
                for j in range(class_num):
                    if int(class_list[j])==1:
                        # 每个词在某个类出现的频率
                        word_appear_set[i][j] += content.count(voca[i])
                        word_doc_set[i][j] += 1
        for j in range(class_num):
            if int(class_list[j]) == 1:
                class_word_appear_set[j] += len(content)
                doc_class_set[j] += 1
    end = time.time()
    print("time is:",(end-begin))
    #word_appear_set为每个词在某个类中出现的频率  class_word_appear_set类中总词数  word_doc_set每个类出现该词的文档数    doc_class_set类中总文档数
    #顺序就是词的顺序
    temp = []
    for i in range(len(voca)):
        temp.append([word_appear_set[i],class_word_appear_set,word_doc_set[i],doc_class_set])
    ans = pd.DataFrame(temp,index=voca,columns=['word_appear_set','class_word_appear_set','word_doc_set','doc_class_set'])
    print(ans)
    return ans