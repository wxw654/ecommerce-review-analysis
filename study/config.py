class Config():
    def __init__(self):
        # 从 study/config.py 出发，先上一层到04_code，再进入01-data/data
        self.train_path = "../01-data/data/train.txt"
        self.test_path = "../01-data/data/test.txt"
        self.dev_path = "../01-data/data/dev.txt"
        self.class_path = "../01-data/data/class.txt"

if __name__ == '__main__':
    conf = Config()
    print(f"训练集路径: {conf.train_path}")
    print(f"类别文件路径: {conf.class_path}")

