import argparse
from importlib import import_module as 导入模块

解析器 = argparse.ArgumentParser(description='中文文本分类器')
解析器.add_argument('--模型文件名', type=str, required=True,
                 help='选择一个模型：TextCNN, 文本循环神经网络, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
解析器.add_argument('--词嵌入方式', default='预处理', type=str, help='随机或者预载训练集')
解析器.add_argument('--word', type=bool, default=False, help='真为词，假为字符')
参数 = 解析器.parse_args()

if __name__ == '__main__':
    数据文件夹 = '清华中文文本分类工具包'
    """
        # 它是指单个词的具体数值矩阵；
        # 这个矩阵内用数字表示着这个词；
        # 这里是加载别已经处理出词的数值矩阵
    """
    词向量文件名 = '搜狗新闻词向量.npz'

    模型文件名 = 参数.模型文件名
    if 模型文件名 == 'FastText':
        # from utils_fast
        参数.词嵌入方式 = '随机'
    else:
        pass

    模型 = 导入模块('模型库.' + 模型文件名)
    模型配置 = 模型.配置(数据文件夹, 词向量文件名, 参数.词嵌入方式)
