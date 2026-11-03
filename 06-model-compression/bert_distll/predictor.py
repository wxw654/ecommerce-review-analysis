# 1、导包
import time
import torch
from bilstm_classifier import BiLSTMClassifier
from config import Config

# 2、初始化配置
conf = Config()
device = conf.device
tokenizer = conf.tokenizer

# 3、保存模型实例
_model = None


# 4、加载模型
def load_model():
    # 使用文件顶部的全局变量
    global _model

    # 1、首次调用时加载模型
    if _model is None:
        print("加载模型...")

        model = BiLSTMClassifier().to(device)

        # 软标签蒸馏模型
        model_path = conf.distill_s_model_save_path

        # 硬标签蒸馏模型：需要时替换上面的路径
        # model_path = conf.distil_h_model_save_path

        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

        # 加载成功后保存
        _model = model

    # 2、返回已加载的模型
    return _model


# 5、预测文本
def predict(data):
    # 1、获取文本
    text = data["text"]

    # 空文本不执行预测，仍返回相同结构的字典
    if not text.strip():
        return {
            "text": text,
            "pred_class": None,
            "total_time": 0.0,
        }

    # 2、分词并转为 Tensor
    encoded = tokenizer(
        text,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # 3、获取模型
    model = load_model()

    # 4、执行预测
    start_time = time.time()

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

        # 5、获取类别索引
        pred_id = torch.argmax(logits, dim=1).item()

    # 6、获取类别名称和耗时（毫秒）
    total_time = (time.time() - start_time) * 1000
    pred_class = conf.class_list[pred_id]

    # 7、返回结果
    return {
        "text": text,
        "pred_class": pred_class,
        "total_time": total_time,
    }


# 6、测试预测
if __name__ == "__main__":
    sample_data = {
        "text": "中华女子学院：本科层次仅1专业招男生"
    }

    result = predict(sample_data)

    print(f"预测文本：{result['text']}")
    print(f"预测结果：{result['pred_class']}")
    print(f"预测耗时：{result['total_time']:.2f}ms")
