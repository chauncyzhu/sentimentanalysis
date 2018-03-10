# coding=gbk
import utils.sentiment_data_path as sdp
import utils.read_data as rd
import pandas as pd
import utils.change_data as change
"""
    数据导入，并取出训练集和测试集
"""
def getTrainAndTest(pos_file,neg_file):
    """
    每个类中抽取20%作为测试集，注意这里的特殊性，因为读取的是已经分好词的文件，所以需要用eval进行转换
    :param pos_file:
    :param neg_file:
    :return:
    """
    pd_pos_data = rd.read_csv(pos_file) #读取所有数据并转为pandas dataframe
    pd_neg_data = rd.read_csv(neg_file)

    pd_train = pd_pos_data.head(int(len(pd_pos_data)*0.8)).append(pd_neg_data.head(int(len(pd_neg_data)*0.8)),ignore_index=True)  #训练集占80%
    pd_test = pd_pos_data.tail(len(pd_pos_data) - int(len(pd_pos_data)*0.8)).append(pd_neg_data.tail(len(pd_neg_data) - int(len(pd_neg_data)*0.8)), ignore_index=True)  # 测试集占20%

    pd_train['content'] = pd_train['content'].apply(change.getOriginalValue)
    pd_train['class'] = pd_train['class'].apply(change.getOriginalValue)
    pd_test['content'] = pd_test['content'].apply(change.getOriginalValue)
    pd_test['class'] = pd_test['class'].apply(change.getOriginalValue)

    print(pd_train)
    print(pd_test)
    print(pd_train['description'],pd_train["content"])
    return (pd_train,pd_test)

if __name__ == '__main__':
    pos_file = sdp.POS_COMMENT
    neg_file = sdp.NEG_COMMENT
    getTrainAndTest(pos_file, neg_file)
