# 1  导包
from bert_classifier import BertClassifier  # 原始模型的类
import torch  # 加载模型、具体量化
from utils import build_dataloader  # 加载数据
from train import model2dev  # 完成评估
from config import Config

# 2  加载配置 和 设备信息
conf = Config()
# todo 动态量化 生成的是 torch cpu 量化模块，它依赖 cpu int8 后端
#  cuda gpu 没有对应的动态量化执行内核
device = torch.device("cpu")

# 3  加载模型
# 初始化模型对象
model = BertClassifier()
# 加载状态字典
model.load_state_dict(torch.load(conf.model_save_path, map_location=device))
model.eval()

# 4  加载数据，评估 量化前的模型
# _, _, dev_loader = build_dataloader()
# report, f1score, acc, precision = model2dev(model, dev_loader, device)
# print(f"量化前模型的分类报告:\n{report}")
# print(f"量化前模型的F1:\n{f1score}")
# print(f"量化前模型的准确率:\n{acc}")
# print(f"量化前模型的精确率:\n{precision}")

# 5  具体量化操作
# todo {torch.nn.Linear} 这里仅对线性层量化
#  常见的可以 DQ 量化的层：Linear、RNN、LSTM、GRU、Conv2d
#  不可以做量化的层：Norm、Embedding，这些层对数值非常敏感，容易引起精度严重衰退
#  不需要对 Dropout 做量化，model.eval() 直接跳过这个层
#  dtype=torch.qint8 量化后的数据类型，其他选择：quint8 无符号整型、float16...
#  qin8 quantized int8 也就是量化后的 8 位整数类型
quantized_model = torch.quantization.quantize_dynamic(
    model,  # 指定量化的模型
    {torch.nn.Linear},  # 指定量化的层的类型，不要加 括号()
    dtype=torch.qint8,  # 量化为的数据类型，
)

# 6  评估 量化后的模型
# report, f1score, acc, precision = model2dev(quantized_model, dev_loader, device)
# print(f"量化后模型的分类报告:\n{report}")
# print(f"量化后模型的F1:\n{f1score}")
# print(f"量化后模型的准确率:\n{acc}")
# print(f"量化后模型的精确率:\n{precision}")

# 7  打印量化后模型的存储占用
# model.parameters()  # 模型的参数
# p.numel() 一组参数的元素总数
# p.element_size() 每个元素占用的字节大小
# sum(p.numel() * p.element_size() for p in quantized_model.parameters()) 模型占用存储的大小，单位 B
# B -》 MB 要除以 1024 ** 2
total = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
print(f"量化前的模型占用存储为:\t{total}MB")

q_total = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024 ** 2
print(f"量化后的模型占用存储为:\t{q_total}MB")

# todo 打印结果 64 MB，在目录中实际显示 145 MB，原因如下：
#  量化的部分，已经不在 parameters() 中，单独保存为了 tuple 元组类型，不再是 tensor
state_dict = quantized_model.state_dict()
for k, v in state_dict.items():
    print(k, " ❤️ ", type(v))

# 8  模型保存
# 两种方式：整个模型 or 状态字典
# model.state_dict()
# torch.save(
#     quantized_model.state_dict(),  # 量化后模型的状态字典
#     conf.quantized_model_state_dict_save_path  # 保存路径
# )

# torch.save(
#     quantized_model,  # 整个模型
#     conf.quantized_model_save_path  # 保存路径
# )
print("量化模型保存成功！")
