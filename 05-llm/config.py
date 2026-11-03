class Config(object):
    def __init__(self):
        # 原始数据路径
        self.train_path = "../01-data/data/train.txt"
        self.test_path = "../01-data/data/test.txt"
        self.dev_path = "../01-data/data/dev.txt"
        self.class_path = "../01-data/data/class.txt"


if __name__ == '__main__':
    conf = Config()
    print(conf.train_path)
    print(conf.test_path)
