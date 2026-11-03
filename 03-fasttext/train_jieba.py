# 1  导包
import fasttext
from config import Config
import datetime
import os

# 2  获取配置信息、时间、数据集路径
conf = Config()
current_time = datetime.datetime.now().date().today().strftime("%m%d")
TRAIN_PATH = "./data/train_ft_jieba.txt"  # 训练集路径
TEST_PATH = "./data/test_ft_jieba.txt"  # 测试集路径


# 3  默认参数训练
def train_default():
    # 1  模型训练
    # fasttext.train_unsupervised()  # 训练词向量 无监督
    model = fasttext.train_supervised(input=TRAIN_PATH)
    print("模型训练完成")

    # 2  模型保存
    # 拼接模型保存路径
    path = os.path.join(conf.ft_model_save_path, f"ft_jieba_{str(current_time)}.bin")
    # 做一步适配 win \ linux /
    path = os.path.normpath(path)  # 统一了连接符 \
    model.save_model(path=path)

    # 3  模型预测
    sent = "词汇 阅读 是 关键   08 年 考研 暑期 英语 复习 全 指南"
    print(sent)
    print(model.predict(sent))

    # 4  查看 词表、近义词、单词向量
    print(f"模型词表前 50 个 token:\n{model.words[:50]}")
    print(f"'游戏'的邻近词:\t{model.get_nearest_neighbors('游戏')}")
    print(f"'游戏'的词向量:\t{model.get_word_vector('游戏')}")

    # 5  模型测试
    result = model.test(path=TEST_PATH)
    print(f"测试结果为:\n{result}")


# 4  自动调参训练
def train_autotune():
    # 1  模型训练
    # fasttext.train_unsupervised()  # 训练词向量 无监督
    model = fasttext.train_supervised(
        input=TRAIN_PATH,  # 训练集
        autotuneValidationFile=TEST_PATH,  # 指定验证集，因为模型训练后要评估
        autotuneDuration=60,  # 自动搜索的时间，单位 s
        epoch=30,  # 不宜太小，否则效果不好
        # dim=128,  # 指定词嵌入维度
        loss="hs",  # 层次 softmax，目的是加速训练
        thread=10,  # 线程数
        verbose=3,  # 常用 2 or 3，输出日志的打印级别，越大越详细
    )
    print("模型训练完成")

    # 2  模型保存
    # 拼接模型保存路径
    path = os.path.join(conf.ft_model_save_path, f"ft_jieba_{str(current_time)}.bin")
    # 做一步适配 win \ linux /
    path = os.path.normpath(path)  # 统一了连接符 \
    model.save_model(path=path)

    # 3  模型预测
    sent = "词汇 阅读 是 关键   08 年 考研 暑期 英语 复习 全 指南"
    print(sent)
    print(model.predict(sent))

    # 4  查看 词表、近义词、单词向量
    print(f"模型词表前 50 个 token:\n{model.words[:50]}")
    print(f"'游戏'的邻近词:\t{model.get_nearest_neighbors('游戏')}")
    print(f"'游戏'的词向量:\t{model.get_word_vector('游戏')}")

    # 5  模型测试
    result = model.test(path=TEST_PATH)
    print(f"测试结果为:\n{result}")


if __name__ == '__main__':
    train_default()
    train_autotune()
