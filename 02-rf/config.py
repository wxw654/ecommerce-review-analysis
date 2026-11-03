class Config():
    def __init__(self):
        # 1  原始数据路径
        self.train_path = "../01-data/data/train.txt"
        self.test_path = "../01-data/data/test.txt"
        self.dev_path = "../01-data/data/dev.txt"
        self.class_path = "../01-data/data/class.txt"

        # 2  处理好后的数据路径
        self.preprocess_train_path = "./data/preprocess_train.csv"
        self.preprocess_test_path = "./data/preprocess_test.csv"
        self.preprocess_dev_path = "./data/preprocess_dev.csv"

        # 3  停用词路径，最终要传给 TFIDF 转换器（把文本转成向量）
        self.stop_words_path = "../01-data/data/stopwords.txt"

        # 4  模型保存路径
        self.rf_model_save_path = "./saved"  # 两个模型：rf 随机森林；tf-idf 转换器模型
        self.model_predict_result = "./result"  # 模型预测结果 保存路径


if __name__ == '__main__':
    conf = Config()
    print(conf.train_path)
    print(conf.rf_model_save_path)
