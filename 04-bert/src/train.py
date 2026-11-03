# 1  导包
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,  # 分类报告
    f1_score,  # f1 值
    accuracy_score,  # 准确率
    precision_score,  # 精确率
)
from torch.optim import AdamW  # 可选 Adam or AdamW
from tqdm import tqdm

from bert_classifier import BertClassifier  # 模型
from config import Config  # 配置信息
from utils import build_dataloader  # 数据加载器

conf = Config()


# 2  模型训练
def model2train():
    # 训练模型，并保存在验证集上表现最好的权重
    # 一 【4】训练准备：数据、模型、优化器、损失函数
    # 数据
    train_loader, _, dev_loader = build_dataloader()
    # 模型
    model = BertClassifier().to(conf.device)
    # 优化器
    optimizer = AdamW(
        params=model.parameters(),  # 要训练的参数
        lr=conf.learning_rate
    )
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 二 【2】两个循环：内循环（批次）、外循环（轮次），初始化观测值
    best_f1_dev = 0.0  # 历史最佳 F1 值
    num_epochs = conf.num_epochs

    for epoch in range(num_epochs):

        # 开启模型训练模式
        model.train()

        # 其他的观测值的初始化
        total = 0  # 总的样本
        correct = 0  # 预测正确的样本
        total_loss = 0.0  # 总的损失

        # 对 train_loader 做进一步的封装 tqdm 进度条
        # 后续实时信息的展示，也可以放在 tqdm 进度条
        pbar = tqdm(
            train_loader,  # 数据加载器
            desc=f"Epoch {epoch + 1}/{num_epochs}"
        )

        for idx, batch in enumerate(pbar):
            # batch：input_ids, attention_mask, labels
            input_ids, attention_mask, labels = [x.to(conf.device) for x in batch]

            # 三【5】训练的固定五步、统计并打印观测值
            # 1  清空梯度
            optimizer.zero_grad()
            # 2  前向传播
            logits = model(input_ids, attention_mask)
            # 3  计算损失
            loss = loss_fn(logits, labels)
            # 4  反向传播
            loss.backward()
            # 5  更新参数
            optimizer.step()

            # 6  统计计算观测值
            # 总的损失
            total_loss += loss.item()
            # 总的样本
            total += labels.size(0)
            # 预测正确的样本 correct
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

            # 7  显示实时信息
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / (idx + 1):.4f}",
                    "acc": f"{correct / total:.4f}"
                }
            )

        # 四 根据验证集的评估指标，判断是否保存模型
        # 1  评估验证集
        report, f1score, acc, precision = model2dev(model, dev_loader)
        print(f"Dev F1: {f1score:.4f}")

        # 2  保存最优模型
        if f1score > best_f1_dev:
            torch.save(
                model.state_dict(),  # 模型的状态词典
                conf.model_save_path,  # 模型保存路径
            )
            best_f1_dev = f1score  # 更新历史最佳的 F1 值
            print("模型保存成功")


# 3  模型验证
def model2dev(model, data_loader, device=None):  # 边训练 边验证（适用小模型）
    # 返回 评估指标：分类报告、F1值、准确率、精确率
    # 一 【2】 模型、数据
    model.eval()

    # 初始化 结果保存 列表，分别用来保存 预测值、真实值
    dev_preds, dev_labels = [], []

    # 二 【1】 1个循环
    # 关闭梯度，逐批次推理结果，并收集结果
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="模型评估中..."):
            # 把 数据 放到 设备上，并拆包
            input_ids, attention_mask, labels = [x.to(conf.device) for x in batch]

            # 三 【1】 1个推理
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            # preds = tensor( [2,5,7,4,,,] )
            # labels = tensor( [2,6,7,5,,,] )
            dev_preds.append(preds)
            dev_labels.append(labels)

    # 四 统计指标
    # dev_preds = [ tensor1 ( [2,5,7,4,,,] ), tensor2, tensor3 ]
    # dev_labels = [ tensor1 ( [2,5,7,4,,,] ), tensor2, tensor3 ]
    dev_preds = torch.cat(dev_preds).cpu().numpy()
    dev_labels = torch.cat(dev_labels).cpu().numpy()
    # 现在的 dev_preds = [2,5,7,4,,,,,,]
    # 现在的 dev_labels = [2,5,7,4,,,,,,]

    report = classification_report(  # 分类报告
        dev_labels,
        dev_preds,
    )

    acc = accuracy_score(dev_labels, dev_preds)  # 准确率
    f1score = f1_score(dev_labels, dev_preds, average="micro")  # f1值
    precision = precision_score(dev_labels, dev_preds, average="micro")  # 精确率

    return report, f1score, acc, precision


if __name__ == '__main__':
    model2train()
