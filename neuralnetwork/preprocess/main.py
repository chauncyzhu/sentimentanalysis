# coding=gbk
import utils.sentiment_data_path as sdp
import neuralnetwork.preprocess.data_clean.import_data as id
import neuralnetwork.preprocess.voca_dict.voca_data as vd
import neuralnetwork.preprocess.generate_vector.feature as feature
import neuralnetwork.preprocess.generate_vector.transfer_vector as tv
import pandas as pd
"""
    经过预处理后，应产生训练集、测试集、字典，其中训练集和测试集都是序号向量，便于使用wordembedding
"""
def __voca_dict(class_num,pos_file,neg_file,voca_csv=None):
    """
    生成训练集、测试集以及词典
    :param class_num: 一共有多少类别
    :param pos_file:正类文件
    :param neg_file:负类文件
    :param voca_csv: 词典路径，如果存在则写入文件中
    :return: 返回训练集、测试集、词典以及对应的wordembedding
    """

    pd_train, pd_test = id.getTrainAndTest(pos_file,neg_file)  #读取数据并转为dataframe

    # pd_train = pd_train.head(1000)  #控制训练集个数
    # pd_test = pd_test.head(200)    #控制测试集个数

    voca_dict = vd.getRelativeValue(pd_train, vd.getUniqueVocabulary(pd_train),
                                    class_num)  # getUniqueVocabulary比较耗时，存储在csv中

    # 如需增加更多term weighting schema，在这里添加
    feature.getBDCVector(voca_dict, class_num, "bdc")  # 根据字典计算BDC值，需要指定index
    feature.getDFBDCVector(voca_dict, class_num, "df_bdc")  # 根据字典计算DF_BDC值，需要指定index

    if voca_csv:  # 如果存在则写入文件中
        voca_dict.to_csv(voca_csv)

    print(voca_dict)
    return pd_train, pd_test, voca_dict

def __generate_vector(pd_train,pd_test,voca_dict,feature_name,train_csv=None,test_csv=None):
    """
    转换成不同的向量
    :param pd_train:训练集，dataframe类型
    :param pd_test:测试集，dataframe类型
    :param voca_dict:词典，记录了一些内容
    :param feature_name:需要转换的特征
    :param embedding_dim: wordembedding的维度
    :param embedding_csv: wordembedding的维度
    :param train_csv:需要存储的训练集文件
    :param test_csv:需要存储的测试集文件
    :return:无返回，都是传引
    """
    pd_train_copy = pd_train.copy()  #防止数据干扰
    pd_test_copy = pd_test.copy()

    # 测试集和训练集转为向量，在神经网络中使序号向量
    tv.changeToFeatureVector(pd_train_copy, voca_dict, feature_name)
    tv.changeToFeatureVector(pd_test_copy, voca_dict, feature_name)
    if train_csv:
        pd_train_copy.to_csv(train_csv,encoding="utf8")  # 写入训练文件中
    if test_csv:
        pd_test_copy.to_csv(test_csv,encoding="utf8")  # 写入测试文件中

if __name__ == '__main__':
    feature_name = "tf"
    #由于只区分积极、消极，因此只有两个类
    pd_train, pd_test, voca_dict = __voca_dict(2, sdp.POS_COMMENT, sdp.NEG_COMMENT, voca_csv=sdp.VOCA_COMMENT)

    #转换成序号向量
    __generate_vector(pd_train, pd_test, voca_dict,feature_name,train_csv=sdp.TRAIN_COMMENT, test_csv=sdp.TEST_COMMENT)


