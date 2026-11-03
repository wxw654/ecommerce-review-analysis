import time
import warnings

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score
from torch.optim import AdamW
from tqdm import tqdm

from bert_classifier import BertClassifier
from bilstm_classifier import BiLSTMClassifier
from config import Config
from utils import build_dataloader, get_time_diff

warnings.filterwarnings("ignore")  # 忽略警告信息

conf = Config()


def model2train(teacher_model, student_model, train_loader, dev_loader):
    """
    训练学生模型（BiLSTM）使用硬标签蒸馏，学习教师模型（BERT）的预测类别。
    """
    # 1 初始化参数
    best_dev_f1 = 0.0  # 记录最佳验证 F1 分数
    step = 0  # 训练步数计数器
    patience = 3  # 早停耐心值
    epochs_no_improve = 0  # 记录未提升的 epoch 数

    # 2 初始化 优化器 损失函数
    optimizer = AdamW(student_model.parameters(), lr=conf.learning_rate)  # 使用 AdamW 优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，用于硬标签损失

    # 3 遍历每个 epoch
    for epoch in range(conf.num_epochs):
        student_model.train()  # 设置学生模型为训练模式
        teacher_model.eval()  # 设置教师模型为评估模式
        total_loss = 0  # 记录当前 epoch 的总损失
        train_preds, train_labels = [], []  # 记录训练预测和真实标签
        epoch_start_time = time.time()  # 记录 epoch 开始时间

        # 3 遍历训练数据批次
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{conf.num_epochs}")
        for batch in pbar:
            step_start_time = time.time()  # 记录当前 step 开始时间
            batch = [i.to(conf.device) for i in batch]
            input_ids, attention_mask, labels = batch

            # ====================== 填充代码 ===============================

            # 1  拿到 教师模型 的 预测值
            # 教师模型执行推理模式
            with torch.no_grad():
                # teacher_logits.shape = (bs, 10)
                teacher_logits = teacher_model(input_ids, attention_mask)
                # 拿到具体预测值
                # teacher_preds.shape = (bs,)
                teacher_preds = torch.argmax(teacher_logits, dim=-1)

            # 2  拿到 学生模型 的 logits
            # student_logits.shape = (bs, 10)
            student_logits = student_model(input_ids, attention_mask)

            # 3  直接计算交叉熵损失
            loss = criterion(student_logits, teacher_preds)

            # ====================== 填充代码 ===============================

            # 4 剩余训练步骤
            # 梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 5 统计观测值
            total_loss += loss.item()
            preds = torch.argmax(student_logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            step += 1  # 训练步数加 1
            step_duration = time.time() - step_start_time  # 计算 step 耗时

            # 【修改点】在 batch 级别，仅将当前 Loss 和耗时更新到进度条中
            # 移除了原来每 10 个 step 跑一次验证集的极端耗时操作
            avg_loss = total_loss / (step if step > 0 else 1)
            pbar.set_postfix({
                "Step": step,
                "Loss": f"{avg_loss:.4f}",
                "Step_Time": f"{step_duration:.2f}s"
            })

        # 6 计算训练集指标
        train_report = classification_report(train_labels, train_preds)

        # 7 模型验证
        student_model.eval()
        report, f1score, accuracy, precision = model2dev(student_model, dev_loader, conf.device)

        # 8 计算 epoch 耗时
        epoch_duration = time.time() - epoch_start_time

        # 【修改点】将 epoch 级别的验证指标追加到进度条末尾显示
        pbar.set_postfix({
            "Step": step,
            "Loss": f"{avg_loss:.4f}",
            "Dev_F1": f"{f1score:.4f}",
            "Dev_Acc": f"{accuracy:.4f}",
            "Epoch_Time": f"{epoch_duration:.2f}s"
        })

        print(f"\nEpoch {epoch + 1}/{conf.num_epochs}")
        print(f"Epoch Duration: {epoch_duration:.2f}s")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Train 分类报告:\n{train_report}")
        print(f"Dev F1: {f1score:.4f}, Dev Accuracy: {accuracy:.4f}")
        print(f"Dev Precision: {precision:.4f}")
        print(f"Dev 分类报告:\n{report}")

        # 9 保存最佳模型，并检查早停
        if f1score > best_dev_f1:
            best_dev_f1 = f1score
            torch.save(student_model.state_dict(), conf.distill_h_model_save_path)
            print(f">>> 发现更优模型！Dev F1 提升至 {best_dev_f1:.4f}，已保存。")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Dev F1 未提升，当前未提升 epoch 数: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"早停触发！Dev F1 在 {patience} 个 epoch 内未提升，停止训练。")
                break

        student_model.train()


def model2dev(model, data_loader, device):
    # 1 准备工作
    model.eval()
    preds, true_labels = [], []

    # 2 关闭梯度计算
    with torch.no_grad():
        # 3 遍历数据
        for batch in tqdm(data_loader, desc="Evaluating ......"):
            batch = [i.to(device) for i in batch]
            input_ids, attention_mask, labels = batch

            # 4 前向传播
            logits = model(input_ids, attention_mask)

            # 5 预测值
            batch_preds = torch.argmax(logits, dim=1)

            # 6 收集预测值和真实标签
            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        # 7 计算分类报告和指标
        report = classification_report(true_labels, preds)
        accuracy = accuracy_score(true_labels, preds)
        f1score = f1_score(true_labels, preds, average='micro')
        precision = precision_score(true_labels, preds, average='micro')

    return report, f1score, accuracy, precision


if __name__ == '__main__':
    # 记录训练开始时间
    start_time = time.time()

    # 1. 加载配置文件
    conf = Config()

    # 2. 加载数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    # 3. 定义教师模型, 加载模型参数
    teacher_model = BertClassifier().to(conf.device)
    teacher_model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))

    # 4. 定义学生模型
    student_model = BiLSTMClassifier().to(conf.device)

    # 5. 模型训练
    model2train(teacher_model, student_model, train_dataloader, dev_dataloader)

    # 6. 获取训练耗时
    total_training_duration = get_time_diff(start_time)
    print(f"硬标签蒸馏训练耗时： {total_training_duration}")
