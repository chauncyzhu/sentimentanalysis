# coding=utf-8
"""
    传统的基于情感字典的情感分析
"""
class TraditionalSentimentAnalysis():
    negdict = []  # 消极情感词典
    posdict = []  # 积极情感词典
    nodict = []  # 否定词词典
    plusdict = []  # 程度副词词典
    sentiment_text = []  #情感文本，这里应该是已经分好词的，list数据，[[line],[line]]
    predict_sentiment_value = []  #所有数据的预测标签

    #初始化
    def __init__(self,negdict,posdict,nodict,plusdict,sentimen_text,predict_sentiment_value):
        self.negdict = negdict
        self.posdict = posdict
        self.nodict = nodict
        self.plusdict = plusdict
        self.sentiment_text = sentimen_text
        self.predict_sentiment_value = predict_sentiment_value  #注意，这部分需要重新赋值，不然会因为对象引用而共享

    # 基于某种规则来分析微博情感，预测函数，对每一个数据进行分析，私有方法
    def __process_line_sentiment(self,sentimentline):
        p = 0  # 预测的情感得分，负则为消极，正则为积极
        # sd = list(jieba.cut(s))  #这里传入的实际上是已经处理好的句子
        for i in range(len(sentimentline)):
            if sentimentline[i] in self.negdict:
                if i > 0 and sentimentline[i - 1] in self.nodict:
                    p = p + 1
                elif i > 0 and sentimentline[i - 1] in self.plusdict:
                    p = p - 2
                else:
                    p = p - 1
            elif sentimentline[i] in self.posdict:
                if i > 0 and sentimentline[i - 1] in self.nodict:
                    p = p - 1
                elif i > 0 and sentimentline[i - 1] in self.plusdict:
                    p = p + 2
                elif i > 0 and sentimentline[i - 1] in self.negdict:
                    p = p - 1
                elif i < len(sentimentline) - 1 and sentimentline[i + 1] in self.negdict:
                    p = p - 1
                else:
                    p = p + 1
            elif sentimentline[i] in self.nodict:
                p = p - 0.5
        return p

    def processSentimenText(self):
        # datalist = readTextData(datafile)
        number = 0
        for sentence in self.sentiment_text:
            number += 1
            if number % 1000 == 0:
                print("has predict", number, "sentiment")
            self.predict_sentiment_value.append(self.__process_line_sentiment(sentence))

    #true_lable_list长度应该和predict_sentiment_value一样
    def evaluation(self,true_lable_list):
        if len(true_lable_list) != len(self.predict_sentiment_value):
            print("true lable data error---length error")
            return
        count = 0
        for i in range(len(true_lable_list)):
            if true_lable_list[i] * self.predict_sentiment_value[i] > 0:  # truelable>0，表明实际为正，只有temp大于0（这里暂时算上等于0）才是正确的
                count += 1
        accuracy = count * (1.0) / len(self.sentiment_text)
        print("准确率:", accuracy)
        return accuracy