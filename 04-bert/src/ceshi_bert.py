# 1  导包
import torch
# BertModel 加载模型
# BertTokenizer 加载分词器
from transformers import BertModel, BertTokenizer

# 2  加载 bert 专属的分词器 和 bert 模型
tokenizer = BertTokenizer.from_pretrained("../bert-base-chinese")
model = BertModel.from_pretrained("../bert-base-chinese")
model.eval()  # 开启评估模式，关闭 dropout、固定 norm

# 3  测试 分词器
text = "我爱长沙"

# 分词
tokens = tokenizer.tokenize(text)
print(f"分词结果:\n{tokens}")

# 词 ——> token id
input_ids = tokenizer.encode(
    text,  # 原始句子
    return_tensors="pt"  # 返回格式 torch tensor；其他选择 'tf' None（python list）
)
print(f"编码结果:\n{input_ids}")

# 解码回原始文本
# result = tokenizer.decode(input_ids[0])
# print(f"解码结果:\n{result}")
#
# result = tokenizer.decode(
#     input_ids[0],
#     skip_special_tokens=True,  # 跳过特殊符号 cls sep
# )
# print(f"解码结果:\n{result}")

# 4  测试 模型
output = model(input_ids)
# print(output)
last_hidden_state = output.last_hidden_state  # 词向量
pooler_output = output.pooler_output  # 句向量
# (1, 6, 768): 1 句话，这句话有 6 个 token，每个 token 用 768 维的向量表示  -》 词向量
print(last_hidden_state.shape)
# (1, 768): 1 句话，这个句子用一个 768 维的向量表示  -》 句向量
print(pooler_output.shape)
