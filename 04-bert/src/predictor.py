# 1  导包
import torch
from bert_classifier import BertClassifier
from config import Config
import time
from utils import get_time_diff

# 2  配置信息
conf = Config()
tokenizer = conf.tokenizer

# 3  保存模型实例（模型推理的单例模式）
_model = None


# 4  加载模型
def load_model():
    global _model  # 全局变量

    # 1  首次调用需要加载模型
    if _model is None:
        print("加载模型")

        model = BertClassifier().to(conf.device)

        state_dict = torch.load(  # 加载状态词典
            "../saved/best_model.pt",  # 最优模型的保存路径
            map_location=conf.device,  # 模型映射到设备上
        )

        model.load_state_dict(state_dict)  # 加载模型状态词典
        model.eval()

        # 加载成功后并保存
        _model = model

    # 2  返回已经加载的模型
    return _model


# 5  模型推理
def predict(data):  # data {“text”: xxx}
    # 1  获取文本
    text = data["text"]

    # 2  分词 转为 tensor
    tokenized = tokenizer(
        text,  # 需要编码的文本
        return_tensors="pt",  # tf None
    )
    input_ids = tokenized["input_ids"].to(conf.device)  # token id
    attention_mask = tokenized["attention_mask"].to(conf.device)  # 掩码张量

    # 3  获取模型
    model = load_model()

    # 4  执行预测，开启 no_grad 模式
    start_time = time.time()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

        # 直接获取类别索引
        pred_id = torch.argmax(logits, dim=1).item()

    # 5  统计耗时，单位 ms
    total_time = get_time_diff(start_time)
    pred_class = conf.class_list[pred_id]  # str label

    # 6  返回 结果str
    return {
        "text": text,
        "pred_class": pred_class,
        "total_time": total_time
    }


if __name__ == '__main__':
    data = {"text": "中国人民公安大学2012年硕士研究生目录及书目"}
    result = predict(data)
    print(result)
