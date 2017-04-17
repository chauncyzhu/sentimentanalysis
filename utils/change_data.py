# coding=gbk

"""
    数据转变
"""
def getOriginalValue(value):
    """
    对字符串进行转换，注意eval("ac")中ac如果未提前定义则会报错
    :param value: 一个数据，str或其他
    :return: 如果是字符串则进行转换
    """
    if type(value) == str:
        value = eval(value)
        while Ellipsis in value:
            value.remove(Ellipsis)
        return value
    else:
        return value