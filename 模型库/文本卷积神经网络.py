import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class 配置:
    def __init__(self, 数据文件夹, 字向量文件名, 字嵌入方式):
        self.模型名 = '文本卷积神经网络'
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

        self.失活概率 = 0.5
        self.无效改善阈值 = 1000  # 若超过1000批次效果还没提升，则提前结束训练
        self.类别数 = len(self.类别名单)
        self.文字编码字典行数 = 0
        self.需迭代的次数 = 10  # 可能需要改名，暂时无法理解
        self.每批数量 = 128
        self.句子长度 = 32
        self.学习率 = 1e-3
        self.字向量长度 = self.预处理的字向量.size(1) if self.预处理的字向量 is not None else 300
        self.卷积核尺寸 = (2, 3, 4)
        self.卷积核个数 = 256


class 神经网络模型(nn.Module):
    def __init__(self, 模型配置):
        super(神经网络模型, self).__init__()
        if 模型配置.预处理的字向量 is not None:
            self.文字张量 = nn.Embedding.from_pretrained(模型配置.预处理的字向量, freeze=False)
        else:
            self.文字张量 = nn.Embedding(模型配置.文字编码字典行数, 模型配置.字向量长度, padding_idx=模型配置.文字编码字典行数 - 1)

        self.卷积神经网络 = nn.ModuleList([nn.Conv2d(1, 模型配置.卷积核个数, (尺寸, 模型配置.字向量长度)) for 尺寸 in 模型配置.卷积核尺寸])
        self.失活概率 = nn.Dropout(模型配置.失活概率)
        self.全连接层 = nn.Linear(模型配置.卷积核个数 * len(模型配置.卷积核尺寸), 模型配置.类别数)

    def 卷积和池化(self, x, 卷积):
        x = F.relu(卷积(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        输出 = self.文字张量(x[0])
        输出 = 输出.unsqueeze(1)
        输出 = torch.cat([self.卷积和池化(输出, 卷积) for 卷积 in self.卷积神经网络], 1)
        输出 = self.失活概率(输出)
        输出 = self.全连接层(输出)

        return 输出
