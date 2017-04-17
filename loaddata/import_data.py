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
    pd_data = pd.read_csv(filename,index_col=0)
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
    pos = __read_data(sdict.POS_DICT)
    neg = __read_data(sdict.NEG_DICT)
    plus = __read_data(sdict.PLUS_DICT)
    no = __read_data(sdict.NO_DICT)
    print(pos)
    print(neg)
    print(plus)
    print(no)
    # 返回积极和消极数据
    return (pos,neg,plus,no)


if __name__ == '__main__':
    #import_comment()   #pos:10676条  neg:10427
    #import_weibo()    #pos:199574  neg:51743  #看来还是有点少应该把另外两种负面的加进去
    import_sentiment_dict()  #pos:6506  neg:11185  plus:182  no:18

    pass



