# 1  导包
import pandas as pd
import pickle
from config import Config

conf = Config()


# 2  定义评估函数
def evaluate():
    # 1  加载验证集
    dev_df = pd.read_csv(conf.preprocess_dev_path)
    print(dev_df.head())

    # 2  加载模型 转换器
    rf_path = conf.rf_model_save_path + "/rf_model.pkl"
    tfidf_path = conf.rf_model_save_path + "/tfidf_vectorizer.pkl"

    with open(rf_path, "rb") as f:
        model = pickle.load(f)

    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)

    # 3  特征工程：把文本转化为 tfidf
    dev_features = tfidf.transform(dev_df["words"])

    # 4  模型预测
    dev_preds = model.predict(dev_features)
    print(dev_preds)

    # 5  保存结果: dict -> df -> to_csv
    # dict：保存 特征句子 words；预测结果 dev_preds
    dev_dict = {
        "words": dev_df["words"],
        "predictions": dev_preds,
    }
    # 传入 dict 生成 df
    output_df = pd.DataFrame(dev_dict)
    output_path = conf.model_predict_result + "/dev_preds.csv"
    output_df.to_csv(
        output_path,  # 结果的保存路径
        index=False,  # 不需要保存索引
    )
    print("离线文件已经生成")
    print(f"预测文件的前五行为:\n{output_df.head()}")


if __name__ == '__main__':
    evaluate()
