# 1  导包
from config import Config
import pandas as pd
import jieba


# 2  封装 数据处理 函数
def preprocess(input_path, output_path):
    '''
    :param input_path: 输入的数据集路径
    :param output_path: 处理好的数据集保存路径
    '''
    # 1  读取数据
    df = pd.read_csv(
        input_path,  # 数据集路径
        sep="\t",  # 分隔符
        names=["text", "label"]  # 赋值列名
    )
    print(f"前五行数据为:\n{df.head()}")

    # 2  分词处理 apply
    # apply 对每一条文本执行后面的函数
    # jieba.lcut 直接返回 list，等价于 jieba.cut() + list
    # todo 本科未录取还有这些路可以走  ->  本科 未 录取 还有 这些 路 可以 走
    df["words"] = df["text"].apply(lambda text: " ".join(jieba.lcut(text)))
    print(f"处理后的数据为:\n{df.head()}")

    # 3  保存结果
    df.to_csv(
        output_path,  # 保存路径
        index=False,  # 不保存索引列
    )


if __name__ == '__main__':
    conf = Config()

    # 处理训练集
    preprocess(conf.train_path, conf.preprocess_train_path)
    preprocess(conf.test_path, conf.preprocess_test_path)
    preprocess(conf.dev_path, conf.preprocess_dev_path)
