import os.path
import pickle as 泡菜
from tqdm import tqdm as 进度条

字表最大行数 = 10000
"""
    <UNK>: 低频词或未在词表中的词
    <PAD>: 补全字符
    <GO>/<SOS>: 句子起始标识符
    <EOS>: 句子结束标识符
    [SEP]：两个句子之间的分隔符
    [MASK]：填充被掩盖掉的字符
"""
未知字符数, 补全字符数 = '<未知字符数>', '<补全字符数>'


def 构建字表(数据路径, 分割器, 字表行数, 文字至少出现次数=1):
    字表字典 = {}
    with open(数据路径, 'r', encoding='UTF-8') as 文件:
        for 行 in 进度条(文件):
            行 = 行.strip()
            if not 行:
                continue
            句子 = 行.split('\t')[0]
            for 字 in 分割器(句子):
                字表字典[字] = 字表字典.get(字, 0) + 1
        列表 = sorted([_ for _ in 字表字典.items() if _[1] >= 文字至少出现次数], key=lambda x: x[1], reverse=True)[:字表行数]
        字表字典 = {文字字典[0]: 索引 for 索引, 文字字典 in enumerate(列表)}
        字表字典.update({未知字符数: len(字表字典), 补全字符数: len(字表字典) + 1})

    return 字表字典


def 处理数据集(配置, 是否为词语):
    if 是否为词语:
        分割器 = lambda x: x.split(' ')
    else:
        分割器 = lambda x: [y for y in x]

    if os.path.exists(配置.字表路径):
        字表 = 泡菜.load(open(配置.字表路径, 'rb'))
    else:
        字表 = 构建字表(配置.训练集路径, 分割器, 字表最大行数)
        泡菜.dump(字表,open(配置.字表路径,'wb'))
    print(f'字表行数：{len(字表)}')