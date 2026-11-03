# 1  导包
import torch.nn as nn
from transformers import BertModel
from config import Config
from utils import build_dataloader

conf = Config()


# 2  创建模型
class BertClassifier(nn.Module):
    def __init__(self):
        # 初始化模型，模型包含 bert 全连接层
        super().__init__()
        self.bert = BertModel.from_pretrained(conf.bert_path)
        self.fc = nn.Linear(conf.hidden_size, conf.num_classes)

    def forward(self, input_ids, attention_mask, return_hidden=False):
        # input_ids: token id
        # attention_mask：掩码张量，遮盖掉 padding 位置
        # return_hidden=False 后续训练中，是否需要 中间隐层状态(部分知识蒸馏)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # output 有两个值：last_hidden_state（词向量）；pooler_output（句向量）
        pooled = output.pooler_output  # 拿到句向量
        logits = self.fc(pooled)

        if return_hidden:
            return logits, pooled

        return logits


if __name__ == '__main__':
    # 1  构建 数据集加载器
    train_dataloader, _, _ = build_dataloader()

    # 2  取出一个batch
    inputs, mask, labels = next(iter(train_dataloader))

    # 3  创建 模型 bert
    model = BertClassifier()

    # 4  前向传播，得到 logits
    output = model(inputs, mask)

    # 5  计算概率
    probs = nn.functional.softmax(output, dim=-1)

    # 6  打印输出
    print(probs)
    print(probs.shape)  # (batch-size, 10)
