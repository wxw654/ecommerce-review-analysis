# 1  导包
import torch
from torch.utils.data import DataLoader, Dataset
from config import Config
import time
from tqdm import tqdm

conf = Config()


# 工具函数：计算耗时，单位是 ms
def get_time_diff(start_time):
    total_time = (time.time() - start_time) * 1000
    return total_time


# 2  读取原始数据
def load_raw_data(file_path):
    # 1  初始化 data list，用来保存数据
    data = []

    # 2  打开数据文件，逐行遍历
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="正在加载数据..."):
            # 3  去掉空字符，跳过空行
            line = line.strip()
            if not line:
                continue

            # 4  拆分 句子 和 标签
            text, label = line.split("\t")

            # 5  元组，保存到 data
            data.append((text, label))

    # 6  返回 data
    return data


# 3  定义 数据集 类
class TextDataset(Dataset):
    def __init__(self, data):  # 接收数据 data
        self.data = data

    def __len__(self):  # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, idx):  # 根据 idx 拿到一条数据样本
        data = self.data[idx]  # (text, label)
        text = data[0]
        label = int(data[1])
        return text, label


# 4  封装 dataloader
def collate_fn(batch):  # 处理一个 batch 的数据
    # batch: 一个 批次 的数据样本 [(句子1，标签1), (句子2，标签2),,,]
    # 1  分别取出 batch 的 句子 和 标签
    sents = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 2  分词，补齐，token -> id
    encoded = conf.tokenizer(
        sents,  # 指定要编码的一个批次的句子
        padding=True,  # 补齐到本批次最长的句子长度
        return_tensors="pt",  # 直接返回 torch tensor，其他选择 tf(tensorflow tensor) None(python list)
    )

    # 3  分别取出 input_ids attention_mask
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = torch.tensor(labels, dtype=torch.long)  # labels long

    # 4  返回 input_ids attention_mask label
    return input_ids, attention_mask, labels


def build_dataloader():
    # 1  读取数据
    train_data = load_raw_data(conf.train_path)
    test_data = load_raw_data(conf.test_path)
    dev_data = load_raw_data(conf.dev_path)

    # 2  创建 数据集 dataset
    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    dev_dataset = TextDataset(dev_data)

    # 3  封装 数据加载器 loader
    train_loader = DataLoader(
        train_dataset,  # 指定数据集
        batch_size=conf.batch_size,  # 批次大小
        shuffle=True,  # 训练集要打乱
        collate_fn=collate_fn  # 批处理函数：用来处理一个 batch 的数据，补齐句子长度，token -> id
    )
    test_loader = DataLoader(
        test_dataset,  # 指定数据集
        batch_size=conf.batch_size,  # 批次大小
        shuffle=False,  # 测试集不需要打乱
        collate_fn=collate_fn  # 批处理函数：用来处理一个 batch 的数据，补齐句子长度，token -> id
    )
    dev_loader = DataLoader(
        dev_dataset,  # 指定数据集
        batch_size=conf.batch_size,  # 批次大小
        shuffle=False,  # 验证集不需要打乱
        collate_fn=collate_fn  # 批处理函数：用来处理一个 batch 的数据，补齐句子长度，token -> id
    )

    return train_loader, test_loader, dev_loader


if __name__ == '__main__':
    # data = load_raw_data(conf.train_path)
    # print(data[:3])

    train_loader, test_loader, dev_loader = build_dataloader()

    print(f"训练集的批次数:\t{len(train_loader)}")
    print(f"训练集的样本数:\t{len(train_loader.dataset)}")

    # next(iter(train_loader))  # 拿到一个batch数据
    input_ids, attention_mask, labels = next(iter(train_loader))
    print(input_ids)
    print(input_ids.shape)
    print("❤️" * 10)
    print(attention_mask)
    print(attention_mask.shape)
    print(labels)
