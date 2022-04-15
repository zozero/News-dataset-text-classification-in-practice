import time

import numpy as np
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from sklearn import metrics

from 工具函数 import 计算耗费时间


def 初始化神经网络(模型, 方法='xavier', 排除='文字张量', 随机种子=123):
    for 名称, 参数 in 模型.named_parameters():
        if 排除 not in 名称:
            if 'weight' in 名称:
                if 方法 == 'xavier':
                    nn.init.xavier_normal_(参数)
                elif 方法 == 'kaiming':
                    nn.init.kaiming_normal_(参数)
                else:
                    nn.init.normal_(参数)
            elif 'bias' in 名称:
                nn.init.constant_(参数, 0)
            else:
                pass


def 训练(模型配置, 模型, 训练集迭代器, 验证集迭代器, 测试集迭代器, 作家):
    开始时间 = time.time()
    模型.train()
    优化器 = torch.optim.Adam(模型.parameters())
    当前批次 = 0
    验证出最好的损失值 = float('inf')
    当前改善的批次 = 0
    旗帜 = False  # 记录是否很久没有效果提升

    for 当前次数 in range(模型配置.需迭代的次数):
        print('当前次数 [{}/{}]'.format(当前次数 + 1, 模型配置.需迭代的次数))

        for 索引, (训练集样本张量, 标签张量) in enumerate(训练集迭代器):
            输出值张量 = 模型(训练集样本张量)
            模型.zero_grad()
            训练集损失值张量 = F.cross_entropy(输出值张量, 标签张量)
            训练集损失值张量.backward()
            优化器.step()
            if 当前批次 % 100 == 0:
                标签值张量 = 标签张量.data.cpu()
                预测值张量 = torch.max(输出值张量.data, 1)[1].cpu()
                训练集准确率 = metrics.accuracy_score(标签值张量, 预测值张量)
                验证集准确率, 验证集损失值 = 评估(模型配置, 模型, 验证集迭代器)
                if 验证集损失值 < 验证出最好的损失值:
                    验证出最好的损失值 = 验证集损失值
                    torch.save(模型.state_dict(), 模型配置.训练结果保存路径)
                    改善标志 = '*'
                    当前改善的批次 = 当前批次
                else:
                    改善标志 = ''

                耗费时间 = 计算耗费时间(开始时间)
                消息文本 = '迭代器：{0:>6}，训练集损失值：{1:>5.2}，训练集准确率{2:>6.2%}，验证集损失值：{3:>5.2}，验证集准确率{4:>6.2%}，耗费时间：{5} {6}'
                print(消息文本.format(当前批次, 训练集损失值张量.item(), 训练集准确率, 验证集损失值, 验证集准确率, 耗费时间, 改善标志))
                作家.add_scalar('训练集-损失值', 训练集损失值张量.item(), 当前批次)
                作家.add_scalar('验证集-损失值', 验证集损失值, 当前批次)
                作家.add_scalar('训练集-准确率', 训练集准确率, 当前批次)
                作家.add_scalar('验证集-损失值', 验证集准确率, 当前批次)
                模型.train()
            当前批次 += 1
            if 当前批次 - 当前改善的批次 > 模型配置.无效改善阈值:
                print('太长时间无法进一步优化，正在自动停止运行......')
                旗帜 = True
                break
        if 旗帜:
            break
    作家.close()
    测试(模型配置, 模型, 测试集迭代器)


def 测试(模型配置, 模型, 测试集迭代器):
    模型.load_state_dict(torch.load(模型配置.训练结果保存路径))
    模型.eval()
    开始时间 = time.time()
    测试集准确率, 测试集损失值, 测试集报告, 测试集混淆矩阵 = 评估(模型配置, 模型, 测试集迭代器, 是否为测试=True)
    消息文本 = '测试集损失值：{0:>5.2}，测试集准确率：{1:>6.2%}'
    print(消息文本.format(测试集损失值, 测试集准确率))
    print('测试集报告')
    print(测试集报告)
    print('测试集混淆矩阵')
    print(测试集混淆矩阵)
    耗费时间 = 计算耗费时间(开始时间)
    print('耗费时间：', 耗费时间)


def 评估(模型配置, 模型, 数据集迭代器, 是否为测试=False):
    模型.eval()
    损失值总量 = 0
    全部预测值矩阵 = np.array([], dtype=int)
    全部标签值矩阵 = np.array([], dtype=int)
    with torch.no_grad():
        for 句子张量, 标签张量 in 数据集迭代器:
            输出张量 = 模型(句子张量)
            损失值张量 = F.cross_entropy(输出张量, 标签张量)
            损失值总量 += 损失值张量
            标签值矩阵 = 标签张量.data.cpu().numpy()
            预测值矩阵 = torch.max(输出张量.data, 1)[1].cpu().numpy()
            全部标签值矩阵 = np.append(全部标签值矩阵, 标签值矩阵)
            全部预测值矩阵 = np.append(全部预测值矩阵, 预测值矩阵)
    准确率 = metrics.accuracy_score(全部标签值矩阵, 全部预测值矩阵)

    if 是否为测试:
        报告 = metrics.classification_report(全部标签值矩阵, 全部预测值矩阵, target_names=模型配置.类别名单, digits=4)
        混淆矩阵 = metrics.confusion_matrix(全部标签值矩阵, 全部预测值矩阵)
        return 准确率, 损失值总量 / len(数据集迭代器), 报告, 混淆矩阵
    return 准确率, 损失值总量 / len(数据集迭代器)
