# 1  导包
from config import Config
import jieba
import pickle

conf = Config()


# 2  封装预测函数
def predict(data):
    # data 传入的要预测的句子字典
    # 1  加载模型 转换器
    rf_path = conf.rf_model_save_path + "/rf_model.pkl"
    tfidf_path = conf.rf_model_save_path + "/tfidf_vectorizer.pkl"

    with open(rf_path, "rb") as f:
        model = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)

    # 2  分词
    words = " ".join(jieba.lcut(data["text"]))
    print(words)

    # 3  特征工程：向量化
    feature = tfidf.transform([words])

    # 4  模型预测
    y_pred = model.predict(feature)  # [idx] 需要转为 str
    print(y_pred)

    # 5  id -> str label
    # 先拿到 id2class
    # {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education',
    # 4: 'science', 5: 'society', 6: 'politics', 7: 'sports',
    # 8: 'game', 9: 'entertainment'}
    id2class = {
        index: label.strip()
        for index, label in enumerate(open(conf.class_path))
    }
    print(id2class)

    data["pred_class"] = id2class[y_pred[0]]

    # 6  返回 data dict
    return data


if __name__ == '__main__':
    data = {"text": "体验2D巅峰 倚天屠龙记十大创新概览"}
    data = predict(data)
    print(data)
