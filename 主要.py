import argparse
import time
from importlib import import_module as 导入模块

import numpy as np
import torch

解析器 = argparse.ArgumentParser(description='中文文本分类器')
解析器.add_argument('--模型文件名', type=str, required=True,
                 help='选择一个模型：TextCNN, 文本循环神经网络, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
解析器.add_argument('--字嵌入方式', default='预处理', type=str, help='随机或者预载训练集')
解析器.add_argument('--词语', default=False, type=bool, help='真为词语，假为字符')
参数 = 解析器.parse_args()

if __name__ == '__main__':
    数据文件夹 = '清华中文文本分类工具包'
    """
        # 它是指单个字的具体数值矩阵；
        # 这个矩阵内用数字表示着这个字；
        # 这里是加载别已经处理出字的数值矩阵
    """
    字向量文件名 = '搜狗新闻字向量.npz'  # 它是以字为单位的

    模型文件名 = 参数.模型文件名
    if 模型文件名 == 'FastText':
        # from utils_fast
        参数.字嵌入方式 = '随机'
    else:
        from 工具函数 import 处理数据集

    模型 = 导入模块('模型库.' + 模型文件名)
    模型配置 = 模型.配置(数据文件夹, 字向量文件名, 参数.字嵌入方式)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果相同

    开始时间=time.time()
    print("载入数据......")
    处理数据集(模型配置,参数.词语)
