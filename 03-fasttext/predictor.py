# 1  导包
import fasttext
import jieba

# 2  加载模型
model = fasttext.load_model("./saved/ft_jieba_0924.bin")


# 3  定义预测函数
def predict(data):
    # 1  对文本分词 拼接
    words = " ".join(jieba.lcut(data["text"]))
    # print(words)

    # 2  模型预测
    result = model.predict(words)
    # print(result)

    # 3  截取 标签str
    pred_class = result[0][0][9:]
    # print(pred_class)

    # 4  封装进 data，并返回
    data["pred_class"] = pred_class

    return data


if __name__ == '__main__':
    data = {"text": "状元心经：考前一周重点是回顾和整理"}
    print(predict(data))
