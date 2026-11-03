class Config():
    def __init__(self):
        # 1  原始数据路径
        self.train_path = "../01-data/data/train.txt"
        self.test_path = "../01-data/data/test.txt"
        self.dev_path = "../01-data/data/dev.txt"

        # 2  训练模型保存路径
        self.ft_model_save_path = "./saved"

        # 3  样本类别文件
        self.class_data_path = "../01-data/data/class.txt"

        # 4  处理好后的数据路径
        self.final_data = "./data"

        # 5  分词策略
        self.use_char_segmentation = False


if __name__ == '__main__':
    conf = Config()
    print(conf.ft_model_save_path)
    print(conf.use_char_segmentation)
