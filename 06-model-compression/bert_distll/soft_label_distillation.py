import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from bert_classifier import BertClassifier
from bilstm_classifier import BiLSTMClassifier
from config import Config
from hard_label_distillation import model2dev
from utils import build_dataloader, get_time_diff

warnings.filterwarnings("ignore")  # 忽略警告信息
conf = Config()


def model2train():
    # 1 配置参数信息
    T = 2.0  # 温度参数，用于软标签蒸馏
    alpha = 0.7  # 软标签和硬标签损失的权重
    step = 0  # 训练步数计数器
    best_dev_f1 = 0.0  # 记录最佳验证 F1 分数

    # 2 教师训练数据与学生训练数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    # 3 定义教师模型, 加载模型参数
    teacher_model = BertClassifier().to(conf.device)
    teacher_model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))

    # 4 定义学生模型
    student_model = BiLSTMClassifier().to(conf.device)

    # 5 初始化优化器和损失函数
    optimizer = AdamW(student_model.parameters(), lr=conf.learning_rate)  # 使用 AdamW 优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，用于硬标签损失

    # 6 遍历每个 epoch
    for epoch in range(conf.num_epochs):
        # 设置学生模型为训练模式，设置教师模型为评估模式
        student_model.train()
        teacher_model.eval()

        # 7 遍历训练数据批次
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{conf.num_epochs}")
        for batch_index, batch in enumerate(pbar):
            # 8 input_ids, attention_mask, label 放到设备
            batch = [i.to(conf.device) for i in batch]
            input_ids, attention_mask, labels = batch

            # ====================== 填充代码 ===============================

            # 1  获取 教师模型 的 logits 和 预测值
            with torch.no_grad():
                # teacher_logits.shape = (bs, 10)
                # teacher_preds.shape = (bs, )
                teacher_logits = teacher_model(input_ids, attention_mask)
                teacher_preds = torch.argmax(teacher_logits, dim=-1)

            # 2  获取 学生模型 的 logits
            # student_logits.shape = (bs, 10)
            student_logits = student_model(input_ids, attention_mask)

            # 3  计算 KL 散度
            # 3.1  先拿到 概率值 (softmax)
            teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            # 3.2  具体计算 KL 散度
            # logits.shape = (batch-size, num_class)
            # sum 所有元素损失之和
            # mean 所有元素损失之和 / (batch-size * num-class)
            # batchmean 所有元素损失之和 / batch-size
            # todo 问什么乘以 T^2
            #  softmax(logits / T) 梯度计算时 缩小了 T^2 倍，为了梯度量级保持一致，所以这里才这么做
            soft_loss = F.kl_div(
                student_log_probs,  # 学生模型的
                teacher_log_probs,  # 教师模型的
                log_target=True,  # 标记 教师模型的概率 已经做了 log
                reduction="batchmean"  # 在整个 betch 上求平均
            ) * T * T

            # 4  计算 交叉熵 损失
            hard_loss = criterion(student_logits, teacher_preds)

            # 5  融合两种损失
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            # ====================== 填充代码 ===============================

            # 9 训练剩余步骤
            # 梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 10 实时信息
            # 【修改点】在 batch 级别，仅将当前 Loss 更新到进度条中，避免频繁跑验证集拖慢训练
            step += 1
            pbar.set_postfix({
                "Step": step,
                "Loss": f"{loss.item():.4f}"
            })

        # 11 【保留】epoch 级别验证 model2dev
        # 每个 epoch 结束后进行一次完整的验证集评估
        report, f1score, accuracy, precision = model2dev(student_model, dev_dataloader, conf.device)

        # 12 【修改点】将 epoch 级别的验证指标也追加到进度条末尾显示
        pbar.set_postfix({
            "Step": step,
            "Dev_F1": f"{f1score:.4f}",
            "Dev_Acc": f"{accuracy:.4f}",
            "Dev_Pre": f"{precision:.4f}"
        })

        print(f"\nEpoch {epoch + 1}/{conf.num_epochs} 验证集分类报告:\n{report}")

        # 13 保存最佳模型
        if f1score > best_dev_f1:
            best_dev_f1 = f1score
            torch.save(student_model.state_dict(), conf.distill_s_model_save_path)
            print(f">>> 发现更优模型！Dev F1 提升至 {best_dev_f1:.4f}，已保存。")


if __name__ == "__main__":
    start_time = time.time()
    model2train()
    total_training_duration = get_time_diff(start_time)
    print(f"软标签蒸馏训练耗时： {total_training_duration}")
