"""
BERT 全局非结构化剪枝：对所有 encoder 层注意力权重剪枝 30%，L1 范数。
"""
import torch
import torch.nn.utils.prune as prune

from bert_classifier import BertClassifier
from config import Config
from train import model2dev
from utils import build_dataloader

conf = Config()


def compute_sparsity(model):
    """计算所有 encoder 层 query 权重的稀疏度"""
    total_params = 0
    zero_params = 0
    for i in range(12):
        weight = model.bert.encoder.layer[i].attention.self.query.weight
        total_params += weight.numel()
        zero_params += (weight == 0).sum().item()
    return zero_params / total_params if total_params > 0 else 0


def print_weights(weight, name, rows=5, cols=5):
    """打印权重矩阵的前 rows x cols 部分"""
    print(f"\n{name}（前 {rows}x{cols}）：")
    print(weight[:rows, :cols])


def main():
    # 1 加载数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    # 2 加载模型
    model = BertClassifier().to(conf.device)
    model.load_state_dict(torch.load(conf.model_save_path))

    # 3 剪枝前 评估指标
    print("模型剪枝前")
    print(model.bert.encoder.layer[0].attention.self)
    print_weights(
        model.bert.encoder.layer[0].attention.self.query.weight,
        "layer[0].attention.self.query.weight 剪枝前"
    )
    # report, f1score, accuracy, precision = model2dev(model, dev_dataloader, conf.device)
    # print(f"\n剪枝前准确率: {accuracy:.4f}, F1: {f1score:.4f}")

    # ====================== 填充代码 ===============================

    # 1  获取需要剪枝的参数
    # 全局剪枝 所有的编码器层 query 剪枝 30%
    # 初始化一个容器 list，用来保存需要剪枝的参数
    parameters_to_prune = []
    # 因为 bert 有 12 编码器层，逐层循环遍历
    for i in range(12):
        # 获取第 i 层的 query 参数
        query = model.bert.encoder.layer[i].attention.self.query
        # 添加到容器
        parameters_to_prune.append((query, "weight"))

    # 2  具体完成剪枝 【L1范数原则】
    prune.global_unstructured(
        parameters_to_prune,  # 剪枝的对象
        pruning_method=prune.L1Unstructured,  # 根据 L1 范数对哪些小的参数做剪枝
        amount=0.3,  # 剪枝的比例
    )

    # ====================== 填充代码 ===============================

    '''
    主要的剪枝方法类：
        Identity - 不剪枝任何单元，仅生成全为 1 的掩码
        RandomUnstructured - 随机非结构化剪枝，随机选择要剪枝的参数
        L1Unstructured - L1 范数非结构化剪枝，根据 L1 范数最小的参数进行剪枝（常用方法）
        RandomStructured - 随机结构化剪枝，随机剪枝整个通道（channels）（整组参数）
        LnStructured - Ln 范数结构化剪枝，根据 Ln 范数最小的通道进行剪枝，需要指定 n（范数类型）
        CustomFromMask - 自定义掩码剪枝，使用预定义的掩码进行剪枝
        
        非结构化：随机选择参数
        结构化：直接作用于一组参数
    '''

    for module, param in parameters_to_prune:
        prune.remove(module, param)

    # 4 剪枝后 评估指标
    print("\n模型剪枝后")
    print(model.bert.encoder.layer[0].attention.self)
    print_weights(
        model.bert.encoder.layer[0].attention.self.query.weight,
        "layer[0].attention.self.query.weight 剪枝后"
    )
    # report, f1score, accuracy, precision = model2dev(model, dev_dataloader, conf.device)
    # sparsity = compute_sparsity(model)
    # print(f"\n剪枝后准确率: {accuracy:.4f}, F1: {f1score:.4f}\n稀疏度: {sparsity:.4f}")

    # 模型保存
    # torch.save(model.state_dict(), conf.prune_model_save_path)


if __name__ == '__main__':
    main()
