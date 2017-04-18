# coding=gbk
import pandas as pd
import numpy as np
import utils.sentiment_data_path as sdp
import utils.read_data as rd
import utils.change_data as cd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dropout,Dense,Activation
from keras.utils import np_utils  #这个使用还不是很清楚

"""
    通过keras实现lstm神经网络，实际上只要确定好标签、输入、字典，就可以训练该神经网络
"""
def embedding_matrix(vova_csv,embedding_dim,feature_name):
    """
    产生embedding_matrix
    :param vova_value: 词典
    :param embedding_dim: 每个特征的维度，对于tf、bdc等都是1
    :return:
    """
    #从vova_csv中读取文件
    vova_value = rd.read_csv(vova_csv,encoding="gbk")
    print(vova_value)
    embedding_matrix = np.zeros((len(vova_value) + 1, embedding_dim))
    vova_value_list = list(vova_value[feature_name])
    for i in range(len(vova_value)):
        embedding_matrix[i+1] = np.array([vova_value_list[i]])
    return embedding_matrix


def lstm(trainData,trainMark,testData,testMark,embedding_dim,embedding_matrix,maxlen):
    # 填充数据，将每个序列长度保持一致
    trainData = list(sequence.pad_sequences(trainData,maxlen=maxlen,dtype='float64'))  # sequence返回的是一个numpy数组，pad_sequences用于填充指定长度的序列，长则阶段，短则补0，由于下面序号为0时，对应值也为0，因此可以这样
    testData = list(sequence.pad_sequences(testData,maxlen=maxlen,dtype='float64'))  # sequence返回的是一个numpy数组，pad_sequences用于填充指定长度的序列，长则阶段，短则补0

    # 建立lstm神经网络模型
    model = Sequential()  # 多个网络层的线性堆叠，可以通过传递一个layer的list来构造该模型，也可以通过.add()方法一个个的加上层
    #model.add(Dense(256, input_shape=(train_total_vova_len,)))   #使用全连接的输入层
    model.add(Embedding(len(embedding_matrix),embedding_dim,weights=[embedding_matrix],mask_zero=True,input_length=maxlen))  # 指定输入层，将高维的one-hot转成低维的embedding表示，第一个参数大或等于0的整数，输入数据最大下标+1，第二个参数大于0的整数，代表全连接嵌入的维度
    # lstm层，也是比较核心的层
    model.add(LSTM(128))  # 256对应Embedding输出维度，128是输入维度可以推导出来
    model.add(Dropout(0.5))  # 每次在参数更新的时候以一定的几率断开层的链接，用于防止过拟合
    model.add(Dense(1))  # 全连接，这里用于输出层，1代表输出层维度，128代表LSTM层维度可以自行推导出来
    model.add(Activation('sigmoid'))  # 输出用sigmoid激活函数
    # 编译该模型，binary_crossentropy（亦称作对数损失，logloss），adam是一种优化器，class_mode表示分类模式
    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    # 正式运行该模型,我知道为什么了，因为没有补0！！每个array的长度是不一样的，因此才会报错
    X = np.array(list(trainData))  # 输入数据
    print("X:", X)
    Y = np.array(list(trainMark))  # 标签
    print("Y:", Y)
    # batch_size：整数，指定进行梯度下降时每个batch包含的样本数
    # nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次
    model.fit(X, Y, batch_size=16, nb_epoch=10)  # 该函数的X、Y应该是多个输入：numpy list(其中每个元素为numpy.array)，单个输入：numpy.array

    # 进行预测
    A = np.array(list(testData))  # 输入数据
    print("A:", A)
    B = np.array(list(testMark))  # 标签
    print("B:", B)
    classes = model.predict_classes(A)  # 这个是预测的数据
    acc = np_utils.accuracy(classes, B)  # 计算准确率，使用还不是很清楚
    print('Test accuracy:', acc)

if __name__ == '__main__':
    vova_csv = sdp.VOCA_COMMENT
    embedding_dim = 1
    maxlen = 50
    feature_name = "df_bdc"
    #获得embedding矩阵
    embedding_matrix = embedding_matrix(vova_csv, embedding_dim, feature_name)

    #获得训练数据和测试数据
    pd_train = rd.read_csv(sdp.TRAIN_COMMENT,encoding="utf8")
    pd_test = rd.read_csv(sdp.TEST_COMMENT,encoding="utf8")

    def f(x):
        if eval(x) == [1,0]:
            return 1
        else:
            return 0
    pd_train['sequence'] = pd_train['sequence'].apply(cd.getOriginalValue)  #序号向量
    pd_test['sequence'] = pd_test['sequence'].apply(cd.getOriginalValue)

    pd_train['class'] = pd_train['class'].apply(f)
    pd_test['class'] = pd_test['class'].apply(f)

    lstm(list(pd_train['sequence']), list(pd_train['class']), list(pd_test['sequence']), list(pd_test['class']), embedding_dim,embedding_matrix,maxlen)