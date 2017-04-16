# coding=gbk
import pandas as pd
import jieba as jb
import utils.sentiment_data_path as sdata
import utils.sentiment_dict_path as sdict

"""
    数据导入，如txt、csv、excel等，同义转换成pandas dataframe，分好词并存储在csv中
"""
#用pandas读取csv数据并转换成列表格式，注意分词后的内容都放在了content列
def __read_data(filename):
    pd_data = pd.read_csv(filename)
    #对pd_data的每一列数据进行还原
    # for name in pd_data.index:
    #     pd_data[name] = pd_data[name].apply(lambda x:eval(x))
    return pd_data

def import_comment():
    pos_comment = __read_data(sdata.POS_COMMENT)
    neg_comment = __read_data(sdata.NEG_COMMENT)
    print(pos_comment)
    print(neg_comment)
    #返回积极和消极数据
    return (pos_comment,neg_comment)

def import_weibo():
    pos_weibo = __read_data(sdata.POS_WEIBO)
    neg_weibo = __read_data(sdata.NEG_WEIBO)
    print(pos_weibo)
    print(neg_weibo)
    # 返回积极和消极数据
    return (pos_weibo, neg_weibo)

def import_sentiment_dict():
    pass


if __name__ == '__main__':
    import_comment()



