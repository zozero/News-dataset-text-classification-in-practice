import torch


class 数据集迭代器:
    def __init__(self, 数据集, 模型配置):
        self.每批数量 = 模型配置.每批数量
        self.数据集 = 数据集
        self.总批数 = len(self.数据集) // self.每批数量
        self.总批数是否为小数 = len(self.数据集) % self.每批数量 != 0
        self.索引 = 0
        self.设备 = 模型配置.设备

    def _转成张量(self, 数据集):
        x = torch.LongTensor([_[0] for _ in 数据集]).to(self.设备)
        y = torch.LongTensor([_[1] for _ in 数据集]).to(self.设备)
        二元模型 = torch.LongTensor([_[3] for _ in 数据集]).to(self.设备)
        三元模型 = torch.LongTensor([_[4] for _ in 数据集]).to(self.设备)

        句子长度 = torch.LongTensor([_[2] for _ in 数据集]).to(self.设备)

        return (x, 二元模型, 三元模型, 句子长度), y

    def __next__(self):
        if self.总批数是否为小数 and self.索引 == self.总批数:
            数据集 = self.数据集[self.索引 * self.每批数量:len(self.数据集)]
            self.索引 += 1
            return self._转成张量(数据集)
        elif self.索引 > self.总批数:
            self.索引 = 0
            raise StopIteration
        else:
            数据集 = self.数据集[self.索引 * self.每批数量:(self.索引 + 1) * self.每批数量]
            self.索引 += 1
            return self._转成张量(数据集)

    def __iter__(self):
        return self

    def __len__(self):
        if self.总批数是否为小数:
            return self.总批数+1
        else:
            return self.总批数
