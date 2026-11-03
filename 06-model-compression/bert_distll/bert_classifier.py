import torch.nn as nn
from config import Config
from utils import build_dataloader

conf = Config()


class BertClassifier(nn.Module):
    """
    BERT + 全连接层的分类模型。
    """

    def __init__(self):
        """
        初始化模型，包括BERT和全连接层。
        """
        super().__init__()
        self.bert = conf.bert_model
        self.fc = nn.Linear(conf.hidden_size, conf.class_num)

    def forward(self, input_ids, attention_mask, return_hidden=False):
        '''
        bert 有两个输出结果：
                last_hidden_state: 每个 token 对应的向量 (bs, seq-len, hidden_size)  词向量
                pooler_output: cls 对应的向量，用来表示整句话的语义 (bs, hidden_size)  句向量
        '''
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        logits = self.fc(pooled)

        if return_hidden:
            return logits, pooled

        return logits


if __name__ == '__main__':
    # 1、构建数据加载器
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    # 2、取出一个批次
    inputs, mask, labels = next(iter(train_dataloader))

    # 3、创建 BERT 分类模型
    model = BertClassifier()

    # 4、前向计算，得到分类分数
    out = model(inputs, mask)

    # 5、将分数转为各类别的概率
    out = nn.functional.softmax(out, dim=1)

    # 6、打印概率和形状
    print(out)
    print(out.shape)
