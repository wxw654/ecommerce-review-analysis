import torch
from transformers import BertModel, BertTokenizer, BertConfig


class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        # 1 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2 数据路径
        self.data_path = "../../01-data/data"  # 数据集的根路径
        self.train_path = self.data_path + "/train.txt"  # 训练集
        self.dev_path = self.data_path + "/dev.txt"  # 验证集
        self.test_path = self.data_path + "/test.txt"  # 测试集
        self.class_path = self.data_path + "/class.txt"  # 类别文件
        self.class_list = [line.strip() for line in open(self.class_path, encoding="utf-8")]  # 类别名单

        # 3 模型相关
        self.model_name = "bert"  # 模型名称
        self.bert_path = "../bert-base-chinese"  # bert 路径
        self.bert_model = BertModel.from_pretrained(self.bert_path)  # bert 模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # bert 分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path)  # bert 配置
        self.hidden_size = self.bert_config.hidden_size  # bert 词嵌入维度
        self.model_save_path = "../saved/best_model.pt"  # 训练保存路径

        # 4 训练相关
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 2  # epoch数
        self.batch_size = 256  # batch大小
        self.pad_size = 32  # 句子长度
        self.learning_rate = 5e-5  # 学习率

        # 5 量化模型保存路径
        self.quantized_model_save_path = "../saved/bert_quantized.pt"
        self.quantized_model_state_dict_save_path = "../saved/bert_quantized_state_dict.pt"

        # 6 剪枝模型训练结果保存路径
        self.prune_model_save_path = "../saved/pruned_bert.pt"


if __name__ == "__main__":
    conf = Config()
    print(conf.bert_config)

    text = "我爱北京天安门。"

    tokens = conf.tokenizer.tokenize(text)
    token_ids = conf.tokenizer.convert_tokens_to_ids(tokens)

    print("分词结果：", tokens)
    print("Token IDs：", token_ids)
    print("类别列表：", conf.class_list)
