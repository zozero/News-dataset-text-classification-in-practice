import numpy as np
import torch


class 配置:
    def __init__(self, 数据文件夹, 字向量文件名, 字嵌入方式):
        self.模型名 = '文本循环神经网络'
        self.训练集路径 = 数据文件夹 + '/数据仓/训练集.txt'
        self.测试集路径 = 数据文件夹 + '/数据仓/测试集.txt'
        self.验证集路径 = 数据文件夹 + '/数据仓/验证集.txt'
        self.类别名单 = [类别.strip() for 类别 in open(数据文件夹 + '/数据仓/类别名单.txt').readlines()]
        self.文字编码字典路径 = 数据文件夹 + '/数据仓/文字编码字典.pkl'
        self.训练结果保存路径 = 数据文件夹 + '/训练结果仓/' + self.模型名 + '.校验点'
        self.日志路径 = 数据文件夹 + '/日志仓/' + self.模型名
        self.预处理的字向量 = torch.tensor(
            np.load(数据文件夹 + '/数据仓/' + 字向量文件名)["embeddings"].astype("float32")) if 字嵌入方式 != '随机' else None
        self.设备 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.失活 = 0.5
        self.无效改进阈值 = 1000  # 若超过1000批次效果还没提升，则提前结束训练
        self.类别数 = len(self.类别名单)
        self.文字编码字典行数 = 0
        self.迭代数 = 10  # 可能需要改名，暂时无法理解
        self.每批数量 = 128
        self.句子长度 = 32
        self.学习率 = 1e-3
        self.字向量长度 = self.预处理的字向量.size(1) if self.预处理的字向量 is not None else 300
        self.单隐层的长度 = 128
        self.字关联的深度 = 2  # 可能是字与字之间相互关联的深度，就是当前字要关联到之前的第几个字
