import torch.nn as nn
from config import Config
from utils import build_dataloader

conf = Config()


class BiLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=conf.tokenizer.vocab_size,  # 词汇表大小
            embedding_dim=conf.embed_size  # 嵌入维度
        )
        self.lstm = nn.LSTM(
            input_size=conf.embed_size,  # 输入维度
            hidden_size=conf.hidden_size_lstm,  # 隐层维度
            num_layers=conf.num_layers,  # LSTM层数
            bidirectional=True,  # 双向LSTM，输出维度翻倍
            batch_first=True  # 批次优先
        )

        self.fc = nn.Linear(
            in_features=conf.hidden_size_lstm * 2,  # 双向LSTM输出维度
            out_features=conf.class_num  # 分类类别数
        )

        self.dropout = nn.Dropout(p=conf.dropout)  # 丢弃概率

        # 如果做中间层的 mse
        # self.fc_hidden = nn.Linear(
        #     in_features=conf.hidden_size_lstm * 2,
        #     out_features=conf.hidden_size
        # )

    def forward(self, input_ids, attention_mask, return_hidden=False):
        """
            input_ids、attention_mask 两者均来自 bert 分词器 tokenizer
            input_ids (torch.Tensor): token id，维度 [batch_size, seq_len]
            attention_mask (torch.Tensor): 注意力掩码，1表示有效token，0表示padding，维度 [batch_size, seq_len]
        """
        # ====================== 填充代码 ===============================

        # 1  词嵌入
        # embed.shape = (bs, seq_len, embedding-size 词嵌入维度)
        embed = self.embedding(input_ids)

        # 2  掩码（与 mask 做哈德玛积，消除 padding 影响）
        # 新的 attention_mask = (bs, seq_len, 1)
        # 新的 embed.shape = (bs, seq_len, embedding-size 词嵌入维度)
        attention_mask = attention_mask.unsqueeze(-1)
        embed = embed * attention_mask

        # 3  经过 BiLSTM
        # lstm_output.shape = (bs, seq_len, hidden-size * 2)
        lstm_output, _ = self.lstm(embed)

        # 4  取最后一个时间步的 hidden-state，就能代表整个句子，去完成后面的分类任务
        # seq_len 等价于 时间步的步数
        # hidden.shape = (bs, hidden-size * 2)
        hidden = lstm_output[:, -1, :]

        # 5  经过 fc，得到 logits
        # 先经过 随机失活
        # logits.shape = (bs, 10)
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)

        # 6  输出软标签 logits
        return logits

        # ====================== 填充代码 ===============================


if __name__ == '__main__':
    # 1 实例化模型
    model = BiLSTMClassifier()
    print(model)
    # 2 加载数据
    train_loader, _, _ = build_dataloader()
    # next 拿到数据迭代器中的 一个 batch 的数据
    input_ids, attention_mask, labels = next(iter(train_loader))
    # 3 模型预测
    logits = model(input_ids, attention_mask)
    print(logits.shape)  # (bs, 10)
