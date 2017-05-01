# coding=gbk
import pandas as pd
"""
    数据读取的utils类
"""

def read_csv(source_file,encoding=None):
    source = pd.read_csv(source_file,index_col=0,encoding=encoding)
    print(source)
    return source