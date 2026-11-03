# 1  导包
import pickle  # 序列化模型
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,  # 随机搜索参数，抽取指定数量的组合，更高效，网格搜索，尝试所有组合
    StratifiedKFold,  # K 折交叉验证
    train_test_split,
)
# 流水线模型：把 TF-IDF 特征提取 → 随机森林分类 组合成一个整体
from sklearn.pipeline import Pipeline

from config import Config


# 2  （一般）训练函数
def train_rf(train_path, stop_words_path, model_save_path):
    '''
    :param train_path: 训练集路径
    :param stop_words_path: 停用词路径
    :param model_save_path: 模型保存路径
    '''
    # 1  读取数据
    df = pd.read_csv(
        train_path  # 训练集路径
    )
    words = df["words"]  # 文本特征（需要转为 tfidf）
    labels = df["label"]  # 标签

    print(df.head())
    df.info()  # 查看是否有空值，不需要 print

    # 2  划分数据集
    x_train, x_test, y_train, y_test = train_test_split(
        words,
        labels,
        test_size=0.2,  # 指定测试集的比例
        random_state=42,
        stratify=labels,  # 按照标签均匀划分
    )
    print(f"训练集的样本数:\t{len(x_train)}")
    print(f"测试集的样本数:\t{len(x_test)}")

    # 3  转换 TF_IDF 特征，需要停用词
    # 读取停用词文件：
    # todo 提取文本特征时，需要忽略掉的词语名单，这些词语对分类任务没有帮助
    with open(stop_words_path, "r", encoding="utf-8") as f:
        stop_words = f.read().split()  # stop_words list

    # 实例化 TF_IDF 转换器
    transfer = TfidfVectorizer(
        stop_words=stop_words,  # 停用词
        tokenizer=str.split,  # 分词器，按照空字符划分，已经完成 jieba 分词
        token_pattern=None,  # 关闭默认的正则分词规则
        lowercase=False,  # 不自动转小写，避免因为大小写规则，导致停用词不能匹配，举例：RNN rnn
    )
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    print(f"训练集的第1条样本:\n{x_train[0]}")
    print(f"训练集的特征维度:\t{x_train.shape}")
    print(f"测试集的特征维度:\t{x_test.shape}")
    print(f"特征词表（列表）:\n{transfer.get_feature_names_out()}")
    # print(f"特征词表（字典）:\n{transfer.vocabulary_}")

    # 4  创建 RF 模型
    model = RandomForestClassifier(
        random_state=22,
        n_jobs=10,  # 并行训练的线程数
    )

    # 5  模型训练
    print("模型开始训练...")
    model.fit(x_train, y_train)
    print("模型训练完成")

    # 6  预测和评估
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")
    print(f"准确率:\t{acc:.4f}")
    print(f"精确率:\t{precision:.4f}")
    print(f"召回率:\t{recall:.4f}")
    print(f"F1值:\t{f1:.4f}")

    # 7  保存模型和 TF-IDF 转换器
    save_dir = Path(model_save_path)
    rf_save_path = save_dir / "rf_model.pkl"
    tfidf_save_path = save_dir / "tfidf_vectorizer.pkl"

    with open(rf_save_path, "wb") as f:  # rf 模型保存
        pickle.dump(model, f)

    with open(tfidf_save_path, "wb") as f:
        pickle.dump(transfer, f)

    print(f"模型和转换器已经保存到:\t{save_dir}")


# 3  随机搜索 训练函数  ->  效果能提高 6 个百分点
def tune_rf(train_path, stop_words_path, model_save_path):
    # 1  读取数据集 并 划分
    df = pd.read_csv(
        train_path  # 训练集路径
    )
    words = df["words"]  # 文本特征（需要转为 tfidf）
    labels = df["label"]  # 标签

    x_train, x_test, y_train, y_test = train_test_split(
        words,
        labels,
        test_size=0.2,  # 指定测试集的比例
        random_state=42,
        stratify=labels,  # 按照标签均匀划分
    )

    # 2  读取停用词
    with open(stop_words_path, "r", encoding="utf-8") as f:
        stop_words = f.read().split()  # stop_words list

    # 3  构建 Pipeline: 两个模块 tfidf rf
    pipeline = Pipeline([
        (
            "tfidf",  # 命名
            TfidfVectorizer(
                stop_words=stop_words,  # 停用词
                tokenizer=str.split,  # 分词器，按照空字符划分，已经完成 jieba 分词
                token_pattern=None,  # 关闭默认的正则分词规则
                lowercase=False  # 不自动转小写，避免因为大小写规则，导致停用词不能匹配，举例：RNN rnn
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                random_state=22,
                n_jobs=10,  # 并行训练的线程数
            )
        )
    ])

    # 4  设置候选参数
    # 因为封装了 流水线模型，所以参数名称发生变化，rf__n_estimators
    params = {
        "rf__n_estimators": [100, 200, 300],  # 随机森林中 决策树的数量
        "rf__max_depth": [30, 60, None],  # None 表示不限制这个值
        "rf__min_samples_split": [2, 5, 10],  # 节点分裂的最小样本数
        "rf__min_samples_leaf": [1, 2, 4],  # 叶节点最小样本数
        "rf__max_features": ["sqrt", "log2"],  # 每次分裂考虑的候选特征数
        "rf__class_weight": [None, "balanced"],  # 各类别的训练权重，balanced 类别越少，每条样本权重要高
    }

    # 设置交叉验证
    cv = StratifiedKFold(  # 交叉验证
        n_splits=3,  # 划分几份
        shuffle=True,  # 打乱数据集
        random_state=22
    )

    # 组装随机搜索对象
    search = RandomizedSearchCV(
        estimator=pipeline,  # 指定流水线模型
        param_distributions=params,  # 候选参数范围
        n_iter=10,  # 抽取 10 组参数来训练
        scoring="f1_micro",  # 默认使用的是 acc（本质上调用了 model.score(x_test, y_test)）
        cv=cv,  # 交叉验证
        n_jobs=10,  # 最多同时运行 10 个搜索任务
        verbose=2,  # 参数调参过程中，输出训练日志，值越大，日志越详细
        random_state=42,
        refit=False,  # 参数搜索结束后，False 不会自动重训最佳模型，后面我们自己手动训练
    )

    # 5  模型训练
    print("开始随机搜索...")
    search.fit(x_train, y_train)
    print("搜索训练已完成")

    # 获取最优参数 search.best_params_ 字典
    # key 是参数名 例如 rf__max_depth，value 是参数值 例如 3
    # best_params = {
    #     key.replace("rf__", "", 1): value
    #     for key, value in search.best_params_.items()
    # }

    # {'rf__n_estimators': 300, 'rf__min_samples_split': 2, 'rf__min_samples_leaf': 1,
    # 'rf__max_features': 'log2', 'rf__max_depth': None, 'rf__class_weight': None}
    print(f"最佳参数组合为:\n{search.best_params_}")
    print(f"最佳评估指标 F1值 为:\t{search.best_score_:.4f}")

    # RandomForestClassifier(
    #     random_state=22,
    #     n_jobs=10,  # 并行训练的线程数
    # )


if __name__ == '__main__':
    conf = Config()

    # train_rf(
    #     train_path=conf.preprocess_train_path,
    #     stop_words_path=conf.stop_words_path,
    #     model_save_path=conf.rf_model_save_path
    # )

    tune_rf(
        train_path=conf.preprocess_train_path,
        stop_words_path=conf.stop_words_path,
        model_save_path=conf.rf_model_save_path
    )
