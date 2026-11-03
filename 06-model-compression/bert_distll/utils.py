# 1、导包，加载配置
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import Config

conf = Config()


# 2、计算耗时
def get_time_diff(start_time):
    """返回耗时，单位为毫秒。"""
    return (time.time() - start_time) * 1000


# 3、读取文本和标签
def load_raw_data(file_path):
    """返回 [(文本, 标签), ...]。"""

    # 1、创建数据列表
    data = []

    # 2、打开文件
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading data"):
            # 3、去掉空字符
            line = line.strip()

            # 4、跳过空行
            if not line:
                continue

            # 5、拆分文本和标签
            text, label = line.rsplit("\t", 1)

            # 6、保存样本，注意要 int(label)
            data.append((text, int(label)))

    # 7、返回数据
    return data


# 4、创建数据集
class TextDataset(Dataset):
    """保存数据，按索引取出样本。"""

    def __init__(self, data):
        # 保存数据
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1、取出文本和标签
        text, label = self.data[idx]

        # 2、返回样本
        return text, label


# 5、处理一个批次
def collate_fn(batch):
    """将一批文本和标签转为 Tensor。"""

    # 1、取出文本和标签
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 2、分词、补齐和截断
    encoded = conf.tokenizer(
        texts,
        padding=True,  # 补齐到本批次最长长度
        return_tensors="pt"  # 返回 Torch Tensor
    )

    # 3、取出 token id
    input_ids = encoded["input_ids"]

    # 4、取出掩码：1 为有效位置，0 为补齐位置
    attention_mask = encoded["attention_mask"]

    # 5、标签转为整数 Tensor
    labels = torch.tensor(labels, dtype=torch.long)

    # 6、返回批次数据
    return input_ids, attention_mask, labels


# 6、构建数据加载器
def build_dataloader():
    """返回训练、测试、验证加载器。"""

    # 1、读取数据
    train_data = load_raw_data(conf.train_path)
    test_data = load_raw_data(conf.test_path)
    dev_data = load_raw_data(conf.dev_path)

    # 2、创建数据集
    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    dev_dataset = TextDataset(dev_data)

    # 3、创建训练加载器，打乱顺序
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 4、创建测试加载器，保持顺序
    test_loader = DataLoader(
        test_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 5、创建验证加载器，保持顺序
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 6、返回加载器
    return train_loader, test_loader, dev_loader


if __name__ == "__main__":
    # 1、记录开始时间
    start_time = time.time()

    # 2、创建加载器
    train_loader, test_loader, dev_loader = build_dataloader()

    # 3、查看样本数和批次数
    print(f"训练集样本数：{len(train_loader.dataset)}")
    print(f"训练集批次数：{len(train_loader)}")

    # 4、取出一个批次
    input_ids, attention_mask, labels = next(iter(train_loader))

    # 5、查看数据形状和标签
    print("❤️" * 10)
    print(f"input_ids 形状：{input_ids.shape}")
    print(f"attention_mask 形状：{attention_mask.shape}")
    print(f"labels 形状：{labels.shape}")
    print(f"标签内容：{labels}")

    # 6、查看耗时
    print(f"总耗时：{get_time_diff(start_time):.2f} 毫秒")
