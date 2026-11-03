# 保存数据集文件路径
class Config():
    def __init__(self):
        self.train_path = "./data/train.txt"
        self.test_path = "./data/test.txt"
        self.dev_path = "./data/dev.txt"
        self.class_path = "./data/class.txt"


if __name__ == '__main__':
    conf = Config()
    print(f"训练集路径：{conf.train_path}")
    print(f"类别文件路径：{conf.class_path}")
