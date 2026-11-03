import torch
from transformers.models import BertModel, BertTokenizer, BertConfig


class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """

        # 数据路径
        self.data_path = "../../01-data/data/"  # 数据集的根路径
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"  # 验证集
        self.test_path = self.data_path + "test.txt"  # 测试集
        self.class_list = [line.strip() for line in open("../../01-data/data/class.txt", encoding="utf-8")]
        self.class_num = len(self.class_list)

        # BERT+FC 原模型训练结果保存路径
        self.model_save_path = "../../04-bert/saved/best_model.pt"

        # BERT+FC 蒸馏模型存储结果路径
        self.distill_model_save_dir = "./saved/"
        self.distill_h_model_save_path = self.distill_model_save_dir + "distill_lstm_h.pt"
        self.distill_s_model_save_path = self.distill_model_save_dir + "distill_lstm_s.pt"

        # 模型训练相关
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 10
        self.batch_size = 256
        self.pad_size = 32
        self.learning_rate = 5e-5

        # BERT相关
        self.bert_path = "../../04-bert/bert-base-chinese"
        self.bert_model = BertModel.from_pretrained(self.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path)
        self.hidden_size = self.bert_config.hidden_size

        # BiLSTM模型参数配置
        self.embed_size = 256
        self.hidden_size_lstm = 512
        self.num_layers = 2
        self.dropout = 0.3
        self.save_model_path = "saved"
        self.dropout = 0.3


if __name__ == '__main__':
    conf = Config()
    print(conf.bert_config)
    print(conf.tokenizer)
