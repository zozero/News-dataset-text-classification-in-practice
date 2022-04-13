import os.path
import pickle as 泡菜
import time
from datetime import timedelta

from tqdm import tqdm as 进度条

文字编码字典最大行数 = 10000
"""
    <UNK>: 低频词或未在词表中的词
    <PAD>: 补全字符
    <GO>/<SOS>: 句子起始标识符
    <EOS>: 句子结束标识符
    [SEP]：两个句子之间的分隔符
    [MASK]：填充被掩盖掉的字符
"""
未知字符数, 补全字符数 = '<未知字符数>', '<补全字符数>'


def 构建文字编码字典(数据路径, 分割器, 文字编码字典行数, 文字至少出现次数=1):
    文字编码字典 = {}
    with open(数据路径, 'r', encoding='UTF-8') as 文件:
        for 行 in 进度条(文件):
            行 = 行.strip()
            if not 行:
                continue
            句子 = 行.split('\t')[0]
            for 字 in 分割器(句子):
                文字编码字典[字] = 文字编码字典.get(字, 0) + 1
        列表 = sorted([_ for _ in 文字编码字典.items() if _[1] >= 文字至少出现次数], key=lambda x: x[1], reverse=True)[:文字编码字典行数]
        文字编码字典 = {文字字典[0]: 索引 for 索引, 文字字典 in enumerate(列表)}
        文字编码字典.update({未知字符数: len(文字编码字典), 补全字符数: len(文字编码字典) + 1})

    return 文字编码字典


def 加载数据集(路径, 分割器, 文字编码字典, 句子长度=32):
    句子列表 = []
    with open(路径, 'r', encoding='UTF-8') as 文件:
        for 行 in 进度条(文件):
            行 = 行.strip()
            if not 行:
                continue
            句子, 句子标签 = 行.split('\t')
            文字标签列表 = []
            文字列表 = 分割器(句子)
            文字列表长度 = len(文字列表)
            if 文字列表长度 < 句子长度:
                文字列表.extend([文字编码字典.get(补全字符数)] * (句子长度 - 文字列表长度))
            else:
                文字列表 = 文字列表[:句子长度]
                文字列表长度 = 句子长度

            for 字 in 文字列表:
                句子列表.append(文字编码字典.get(字, 文字编码字典.get(未知字符数)))
            句子列表.append((文字标签列表, int(句子标签), 文字列表长度))

    return 句子列表


def 处理数据集(配置, 是否为词语):
    if 是否为词语:
        分割器 = lambda x: x.split(' ')
    else:
        分割器 = lambda x: [y for y in x]

    if os.path.exists(配置.文字编码字典路径):
        文字编码字典 = 泡菜.load(open(配置.文字编码字典路径, 'rb'))
    else:
        文字编码字典 = 构建文字编码字典(配置.训练集路径, 分割器, 文字编码字典最大行数)
        泡菜.dump(文字编码字典, open(配置.文字编码字典路径, 'wb'))
    print(f'文字编码字典行数：{len(文字编码字典)}')

    训练集 = 加载数据集(配置.训练集路径, 分割器, 文字编码字典, 配置.句子长度)
    验证集 = 加载数据集(配置.验证集路径, 分割器, 文字编码字典, 配置.句子长度)
    测试集 = 加载数据集(配置.测试集路径, 分割器, 文字编码字典, 配置.句子长度)

    return 文字编码字典, 训练集, 验证集, 测试集


def 计算耗费时间(开始时间):
    结束时间 = time.time()
    耗费时间 = 结束时间 - 开始时间
    return timedelta(seconds=int(耗费时间))
