import argparse
import time
from importlib import import_module as 导入模块
from tensorboardX import SummaryWriter

# from 模型库 import 文本重叠神经网络 as 模型

import numpy as np
import torch

from 数据集迭代器类 import 数据集迭代器
from 训练与评估函数 import 初始化神经网络, 训练

解析器 = argparse.ArgumentParser(description='中文文本分类器')
解析器.add_argument('--模型文件名', type=str, required=True,
                 help='选择一个模型：文本卷积神经网络, 文本重叠神经网络, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
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
        from 工具函数 import 处理数据集, 计算耗费时间

    模型 = 导入模块('模型库.' + 模型文件名)
    模型配置 = 模型.配置(数据文件夹, 字向量文件名, 参数.字嵌入方式)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果相同

    开始时间 = time.time()
    print("载入数据......")
    文字编码字典, 训练集, 验证集, 测试集 = 处理数据集(模型配置, 参数.词语)
    训练集迭代器 = 数据集迭代器(训练集, 模型配置)
    验证集迭代器 = 数据集迭代器(验证集, 模型配置)
    测试集迭代器 = 数据集迭代器(测试集, 模型配置)
    耗费时间 = 计算耗费时间(开始时间)
    print('耗费时间', 耗费时间)

    模型配置.文字编码字典行数 = len(文字编码字典)
    神经网络模型 = 模型.神经网络模型(模型配置).to(模型配置.设备)
    作家 = SummaryWriter(log_dir=模型配置.日志路径 + '/' + time.strftime("%m-%d_%H.%M", time.localtime()))
    if 模型文件名 != 'Transformer':
        初始化神经网络(神经网络模型)

    print(神经网络模型.parameters)

    训练(模型配置, 神经网络模型, 训练集迭代器, 验证集迭代器, 测试集迭代器, 作家)
