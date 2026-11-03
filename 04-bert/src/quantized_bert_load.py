# 1  导包
import torch
# 模型类的定义
from bert_classifier import BertClassifier
from config import Config
import transformers

conf = Config()


# 2  直接加载模型
def load_model():
    # 把 自己信任的 类，添加到白名单
    torch.serialization.add_safe_globals([BertClassifier])
    torch.serialization.add_safe_globals([transformers.models.bert.modeling_bert.BertModel])
    torch.serialization.safe_globals([transformers.models.bert.modeling_bert.BertEmbeddings])

    model = torch.load(
        conf.quantized_model_save_path,
        # False 加载整个模型（自己保存的信任模型）
        weights_only=True
    )
    model.eval()
    print("模型加载成功")


# 3  加载模型的状态词典
def load_model_state_dict():
    # 1  实例化模型对象
    model = BertClassifier()
    model.eval()

    # 2  需要对空模型做动态量化（与之前要一模一样）
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # 指定量化的模型
        {torch.nn.Linear},  # 指定量化的层的类型，不要加 括号()
        dtype=torch.qint8,  # 量化为的数据类型，
    )

    # 3  加载状态字典
    quantized_model.load_state_dict(
        torch.load(
            conf.quantized_model_state_dict_save_path  # 状态字典的保存路径
        )
    )

    # 4  设置评估模式
    quantized_model.eval()
    print("模型加载成功")


if __name__ == '__main__':
    load_model()
    # load_model_state_dict()
